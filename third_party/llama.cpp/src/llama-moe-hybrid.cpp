#include "llama-moe-hybrid.h"

#ifdef LLAMA_LAYER_STREAMING

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define MOE_HYBRID_PAUSE() _mm_pause()
#else
#define MOE_HYBRID_PAUSE() std::this_thread::yield()
#endif

// T-Q36-AER experimental hybrid MoE decode, host side.
//
// Env (read once when the engine is created):
//   LLAMA_MOE_HYBRID          VRAM cache slots per MoE layer (> 0 enables)
//   LLAMA_MOE_HYBRID_FREQ     required: n_layer lines x n_expert expert counts (train split), used for the
//                             initial cache fill and as the EMA tie-break
//   LLAMA_MOE_HYBRID_B        max promotions per token (default 16, 0 = static cache, < 0 = unlimited)
//   LLAMA_MOE_HYBRID_LAMBDA   EMA decay (default 0.97)
//   LLAMA_MOE_HYBRID_THREADS  host engine threads (default 8)
//   LLAMA_MOE_HYBRID_FIRST    first logical CPU of the engine (default 0)
//   LLAMA_MOE_HYBRID_STRIDE   logical CPU stride between engine threads (default 2)
//   LLAMA_MOE_HYBRID_POLL     engine threadpool poll level (default 100)
//   LLAMA_MOE_HYBRID_LOG      optional JSONL path, one record per decoded token
//
// Policy (identical to build/q36-aer-verify/sim_policy.py "ema"): per layer the score of an expert is its
// routing EMA plus a 1e-6 frequency tie-break. Between tokens the up-to-8 best uncached experts are paired
// with the 8 worst cached ones; each pair with a positive score gain is a promotion candidate, and the B
// largest gains over all layers are promoted. A promoted expert takes its victim's slot: the victim leaves
// the published map at once, the newcomer is published only after its DMA (auxiliary stream) completed.
// The engine thread submits a token's promotion copies right after it answers that token's first host
// request: submitted earlier, the copies would hold the copy engine while the graph's inputs upload.

namespace {

using moe_clock = std::chrono::steady_clock;

constexpr int MOE_HYBRID_K = 8; // candidate pairs per layer and token

static double moe_hybrid_us(moe_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(moe_clock::now() - t0).count();
}

static int moe_hybrid_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static double moe_hybrid_env_double(const char * name, double def) {
    const char * v = getenv(name);
    return v && *v ? strtod(v, nullptr) : def;
}

struct moe_hybrid_host_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor  * x   = nullptr; // F32 [n_embd, 1]
    ggml_tensor  * ids = nullptr; // I32 [m, 1]
    ggml_tensor  * w   = nullptr; // F32 [1, m, 1]
    ggml_tensor  * out = nullptr; // F32 [n_embd, 1]
};

struct moe_hybrid_promo {
    int l;
    int e;
    int s;
};

struct moe_hybrid_pair {
    double gain;
    int    l;
    int    cand;
    int    victim;
};

struct moe_hybrid_engine {
    const llama_model * model = nullptr;
    ggml_backend_t      cuda  = nullptr;

    int n_layer  = 0;
    int n_expert = 0;
    int n_used   = 0;
    int n_slot   = 0;
    int n_embd   = 0;

    int    budget     = 48;
    double lambda     = 0.97;
    int    n_threads  = 8;
    int    cpu_first  = 0;
    int    cpu_stride = 2;
    int    poll       = 100;

    // model host expert tensors per layer
    std::vector<ggml_tensor *> h_up, h_gate, h_down;
    std::vector<const void *>  registered;

    // VRAM cache per layer [ne0, ne1, n_slot + 1]; slot n_slot stays zero and absorbs the misses
    ggml_context *             cache_ctx = nullptr;
    ggml_backend_buffer_t      cache_buf = nullptr;
    std::vector<ggml_tensor *> c_up, c_gate, c_down;
    ggml_tensor *              state = nullptr; // I32[16]: [0] = publish sequence

    // pinned, device-mapped mailbox + routing trace [n_layer][MAX_USED]
    ggml_backend_buffer_t     host_buf  = nullptr;
    ggml_moe_hybrid_mailbox * mb        = nullptr;
    int32_t *                 trace     = nullptr;
    void *                    mb_dev    = nullptr;
    void *                    trace_dev = nullptr;

    // policy state
    std::vector<int32_t>          map;         // [n_layer*n_expert] published slot or -1 (graph input)
    std::vector<int32_t>          slot_of;     // [n_layer*n_expert] slot incl. pending, -1 if not resident
    std::vector<int32_t>          slot_expert; // [n_layer*n_slot]
    std::vector<double>           ema;
    std::vector<double>           tie;
    std::vector<moe_hybrid_promo> pending;
    std::vector<moe_hybrid_pair>  pairs;
    std::vector<void *>           dma_dst;
    std::vector<const void *>     dma_src;
    std::vector<size_t>           dma_size;
    std::atomic<int>              dma_state{0}; // 0 idle, 1 copies queued by on_token, 2 being submitted
    ggml_backend_event_t          ev = nullptr;
    bool                          batch_ok = true;
    bool                          down     = false;

    // main-thread stats
    int64_t n_tokens      = 0;
    int64_t n_hits        = 0;
    int64_t n_lookups     = 0;
    int64_t n_promotions  = 0;
    int64_t n_waits       = 0;
    int64_t n_dma_main    = 0; // promotion batches the engine left to on_token (tokens without misses)
    double  on_token_sum  = 0.0;
    double  on_token_max  = 0.0;
    FILE *  log           = nullptr;

    // per-token GPU timeline sums in us (diagnostics only, from ggml_moe_hybrid_mailbox::prof)
    // span: first publish to last wait end; attn: wait end to next publish (attention, router);
    // ovl: publish to wait start (cached experts, shared expert); stall: GPU wait on the host;
    // hserve: host serve time; rt: publish to wait end of layers with misses; ydl: wait end to result
    // add end of layers with misses (host result load); gap: previous
    // token's last wait end to this token's first publish (head, launch, on_token)
    uint64_t prev_end   = 0; // globaltimer at the previous token's last wait end
    int64_t  n_prof     = 0;
    int64_t  n_gap      = 0;
    int64_t  n_ml       = 0; // layers with misses over the profiled tokens
    double   span_sum   = 0.0;
    double   stall_sum  = 0.0;
    double   hserve_sum = 0.0;
    double   rt_sum     = 0.0;
    double   ydl_sum    = 0.0;
    double   gap_sum    = 0.0;
    double   attn_sum   = 0.0;
    double   ovl_sum    = 0.0;

    // host miss engine (own thread; its stats are read only after join)
    std::thread                        th;
    std::atomic<bool>                  stop{false};
    std::atomic<uint32_t>              wake_gen{0};
    std::mutex                         mtx;
    std::condition_variable            cv;
    ggml_backend_t                     cpu = nullptr;
    ggml_threadpool *                  tp  = nullptr;
    std::vector<moe_hybrid_host_graph> graphs; // [n_layer*n_used], graph (l, m) serves m misses of layer l
    int64_t                            n_requests = 0;
    double                             serve_sum  = 0.0;
    double                             serve_max  = 0.0;
    int64_t                            nmiss_hist[GGML_MOE_HYBRID_MAX_USED + 1] = {};
    double                             compute_sum[GGML_MOE_HYBRID_MAX_USED + 1] = {}; // CPU graph time by miss count
    int64_t                            n_dma_engine   = 0;
    double                             dma_submit_sum = 0.0;

    ~moe_hybrid_engine() { shutdown(); }

    void init(const llm_graph_context & g, const llama_model & mdl, ggml_backend_t be, int slots);
    void shutdown();
    void on_token();

    void wake() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            wake_gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_one();
    }

    void queue_expert(int l, int e, int s) {
        const ggml_tensor * src[3] = { h_up[l], h_gate[l], h_down[l] };
        const ggml_tensor * dst[3] = { c_up[l], c_gate[l], c_down[l] };
        for (int k = 0; k < 3; ++k) {
            dma_dst.push_back((char *) dst[k]->data + (size_t) s*dst[k]->nb[2]);
            dma_src.push_back((const char *) src[k]->data + (size_t) e*src[k]->nb[2]);
            dma_size.push_back(src[k]->nb[2]);
        }
    }

    void submit_dma() {
        if (dma_dst.empty()) {
            return;
        }
        if (!batch_ok || !ggml_backend_cuda_copy_batch_aux(cuda, dma_dst.data(), dma_src.data(), dma_size.data(), dma_dst.size())) {
            if (batch_ok) {
                LLAMA_LOG_WARN("%s: batched copy unavailable, using one async copy per region\n", __func__);
                batch_ok = false;
            }
            for (size_t i = 0; i < dma_dst.size(); ++i) {
                ggml_backend_cuda_memcpy_1d_aux(cuda, dma_dst[i], dma_src[i], dma_size[i]);
            }
        }
        dma_dst.clear();
        dma_src.clear();
        dma_size.clear();
    }

    std::string build_host_graph(moe_hybrid_host_graph & hg, int l, int m);
    void        run(std::promise<std::string> ready);
    void        serve();
};

// host expert tensors must be plain, contiguous host memory the DMA can read and the CPU can compute on
static void moe_hybrid_check_host_tensor(const ggml_tensor * t, const char * what, int il) {
    if (t == nullptr) {
        GGML_ABORT("moe_hybrid: layer %d has no %s tensor", il, what);
    }
    if (t->data == nullptr || t->buffer == nullptr || t->extra != nullptr || !ggml_is_contiguous(t)) {
        GGML_ABORT("moe_hybrid: layer %d %s is not a plain contiguous host tensor (use --no-repack)", il, what);
    }
    const char * buf = ggml_backend_buffer_name(t->buffer);
    if (strcmp(buf, "CPU") != 0 && strcmp(buf, "CPU_Mapped") != 0 && strcmp(buf, "CUDA_Host") != 0) {
        GGML_ABORT("moe_hybrid: layer %d %s lives in buffer %s; the routed experts must stay in host memory (-ncmoe covering every layer)",
            il, what, buf);
    }
}

void moe_hybrid_engine::init(const llm_graph_context & g, const llama_model & mdl, ggml_backend_t be, int slots) {
    const auto t_init = moe_clock::now();

    model    = &mdl;
    cuda     = be;
    n_layer  = (int) g.n_layer;
    n_expert = (int) g.n_expert;
    n_used   = (int) g.hparams.n_expert_used;
    n_slot   = slots;
    n_embd   = (int) g.n_embd;

    budget     = moe_hybrid_env_int("LLAMA_MOE_HYBRID_B", 16);
    lambda     = moe_hybrid_env_double("LLAMA_MOE_HYBRID_LAMBDA", 0.97);
    n_threads  = moe_hybrid_env_int("LLAMA_MOE_HYBRID_THREADS", 8);
    cpu_first  = moe_hybrid_env_int("LLAMA_MOE_HYBRID_FIRST", 0);
    cpu_stride = moe_hybrid_env_int("LLAMA_MOE_HYBRID_STRIDE", 2);
    poll       = moe_hybrid_env_int("LLAMA_MOE_HYBRID_POLL", 100);

    if (n_used < 1 || n_used > GGML_MOE_HYBRID_MAX_USED || n_embd > GGML_MOE_HYBRID_MAX_EMBD || n_layer > GGML_MOE_HYBRID_MAX_LAYER) {
        GGML_ABORT("moe_hybrid: unsupported shape n_expert_used=%d n_embd=%d n_layer=%d", n_used, n_embd, n_layer);
    }
    if (n_slot > n_expert) {
        GGML_ABORT("moe_hybrid: %d slots exceed %d experts per layer", n_slot, n_expert);
    }
    if (!(lambda >= 0.0 && lambda < 1.0) || n_threads < 1 || cpu_first < 0 || cpu_stride < 1) {
        GGML_ABORT("moe_hybrid: bad config lambda=%g threads=%d first=%d stride=%d", lambda, n_threads, cpu_first, cpu_stride);
    }
    if (cpu_first + (n_threads - 1)*cpu_stride >= GGML_MAX_N_THREADS) {
        GGML_ABORT("moe_hybrid: engine CPU mask exceeds %d CPUs", GGML_MAX_N_THREADS);
    }
    if (g.loras != nullptr && !g.loras->empty()) {
        GGML_ABORT("moe_hybrid: LoRA adapters are not supported");
    }
    if (g.hparams.n_expert_groups > 1) {
        GGML_ABORT("moe_hybrid: expert groups are not supported");
    }
    if (!ggml_backend_cuda_aux_stream_ok(cuda)) {
        GGML_ABORT("moe_hybrid: CUDA auxiliary stream unavailable");
    }

    h_up.resize(n_layer);
    h_gate.resize(n_layer);
    h_down.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & L = mdl.layers[il];
        if (L.ffn_gate_inp == nullptr || L.ffn_gate_up_exps || L.ffn_up_exps_s || L.ffn_gate_exps_s || L.ffn_down_exps_s ||
            L.ffn_up_exps_b || L.ffn_gate_exps_b || L.ffn_down_exps_b || L.ffn_gate_up_exps_b || L.ffn_gate_inp_b || L.ffn_exp_probs_b) {
            GGML_ABORT("moe_hybrid: layer %d has an unsupported MoE tensor set", il);
        }
        if (g.hparams.swiglu_clamp_exp[il] > 1e-6f) {
            GGML_ABORT("moe_hybrid: layer %d uses a clamped SwiGLU", il);
        }
        moe_hybrid_check_host_tensor(L.ffn_up_exps,   "ffn_up_exps",   il);
        moe_hybrid_check_host_tensor(L.ffn_gate_exps, "ffn_gate_exps", il);
        moe_hybrid_check_host_tensor(L.ffn_down_exps, "ffn_down_exps", il);
        h_up[il]   = L.ffn_up_exps;
        h_gate[il] = L.ffn_gate_exps;
        h_down[il] = L.ffn_down_exps;
        for (const ggml_tensor * t : { h_up[il], h_gate[il], h_down[il] }) {
            if (t->ne[2] != n_expert) {
                GGML_ABORT("moe_hybrid: layer %d %s has %lld experts, expected %d", il, t->name, (long long) t->ne[2], n_expert);
            }
        }
    }

    // expert frequencies: initial fill and tie-break
    std::vector<int64_t> freq((size_t) n_layer*n_expert);
    {
        const char * path = getenv("LLAMA_MOE_HYBRID_FREQ");
        if (path == nullptr || *path == '\0') {
            GGML_ABORT("moe_hybrid: LLAMA_MOE_HYBRID_FREQ is required");
        }
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            GGML_ABORT("moe_hybrid: cannot open frequency file %s", path);
        }
        size_t n = 0;
        long long v = 0;
        while (n < freq.size() && fscanf(f, "%lld", &v) == 1) {
            freq[n++] = v;
        }
        const bool extra = fscanf(f, "%lld", &v) == 1;
        fclose(f);
        if (n != freq.size() || extra) {
            GGML_ABORT("moe_hybrid: frequency file %s must hold exactly %d x %d counts", path, n_layer, n_expert);
        }
    }
    const int64_t fmax = *std::max_element(freq.begin(), freq.end());
    if (fmax <= 0 || *std::min_element(freq.begin(), freq.end()) < 0) {
        GGML_ABORT("moe_hybrid: invalid frequency counts");
    }

    // page-lock the host expert ranges so promotions DMA straight from the model's memory
    const auto t_reg = moe_clock::now();
    size_t reg_bytes = 0;
    {
        struct range { uintptr_t lo, hi; };
        std::vector<range> rs;
        const uintptr_t page = 4096;
        for (int il = 0; il < n_layer; ++il) {
            for (const ggml_tensor * t : { h_up[il], h_gate[il], h_down[il] }) {
                if (strcmp(ggml_backend_buffer_name(t->buffer), "CUDA_Host") == 0) {
                    continue; // already pinned
                }
                const uintptr_t lo = (uintptr_t) t->data & ~(page - 1);
                const uintptr_t hi = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(page - 1);
                rs.push_back({ lo, hi });
            }
        }
        std::sort(rs.begin(), rs.end(), [](const range & a, const range & b) { return a.lo < b.lo; });
        std::vector<range> merged;
        for (const range & r : rs) {
            if (!merged.empty() && r.lo <= merged.back().hi) {
                merged.back().hi = std::max(merged.back().hi, r.hi);
            } else {
                merged.push_back(r);
            }
        }
        for (const range & r : merged) {
            char err[256] = {};
            if (!ggml_backend_cuda_register_host_range((const void *) r.lo, r.hi - r.lo, err, sizeof(err))) {
                if (strstr(err, "already") != nullptr) {
                    continue; // the layer-streaming runner pinned this range first; DMA works, and it stays the owner of the pin
                }
                GGML_ABORT("moe_hybrid: host range registration failed (%zu bytes): %s", (size_t) (r.hi - r.lo), err);
            }
            registered.push_back((const void *) r.lo);
            reg_bytes += r.hi - r.lo;
        }
    }
    const double reg_ms = moe_hybrid_us(t_reg)/1e3;

    // VRAM cache
    {
        ggml_init_params ip = { ggml_tensor_overhead()*(3*(size_t) n_layer + 1), nullptr, true };
        cache_ctx = ggml_init(ip);
        if (cache_ctx == nullptr) {
            GGML_ABORT("moe_hybrid: cache context allocation failed");
        }
        c_up.resize(n_layer);
        c_gate.resize(n_layer);
        c_down.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor ** dst[3] = { &c_up[il], &c_gate[il], &c_down[il] };
            const ggml_tensor * src[3] = { h_up[il], h_gate[il], h_down[il] };
            for (int k = 0; k < 3; ++k) {
                ggml_tensor * t = ggml_new_tensor_3d(cache_ctx, src[k]->type, src[k]->ne[0], src[k]->ne[1], n_slot + 1);
                GGML_ASSERT(t->nb[1] == src[k]->nb[1] && t->nb[2] == src[k]->nb[2]);
                ggml_format_name(t, "moe_hybrid.%d.%d", il, k);
                *dst[k] = t;
            }
        }
        state = ggml_new_tensor_1d(cache_ctx, GGML_TYPE_I32, 16);
        ggml_set_name(state, "moe_hybrid.state");
        cache_buf = ggml_backend_alloc_ctx_tensors_from_buft(cache_ctx, ggml_backend_get_default_buffer_type(cuda));
        if (cache_buf == nullptr) {
            GGML_ABORT("moe_hybrid: VRAM cache allocation failed (%d slots per layer)", n_slot);
        }
        ggml_backend_buffer_set_usage(cache_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(cache_buf, 0);
    }

    // mailbox + trace
    {
        const size_t mb_size = GGML_PAD(sizeof(ggml_moe_hybrid_mailbox), 256);
        const size_t tr_size = (size_t) n_layer*GGML_MOE_HYBRID_MAX_USED*sizeof(int32_t);
        host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), mb_size + tr_size);
        if (host_buf == nullptr) {
            GGML_ABORT("moe_hybrid: pinned mailbox allocation failed");
        }
        char * base = (char *) ggml_backend_buffer_get_base(host_buf);
        memset(base, 0, mb_size + tr_size);
        mb    = (ggml_moe_hybrid_mailbox *) base;
        trace = (int32_t *) (base + mb_size);
        std::fill(trace, trace + (size_t) n_layer*GGML_MOE_HYBRID_MAX_USED, -1);
        char * dev = (char *) ggml_backend_cuda_host_device_ptr(base);
        if (dev == nullptr) {
            GGML_ABORT("moe_hybrid: mailbox is not device-mapped pinned memory");
        }
        mb_dev    = dev;
        trace_dev = dev + mb_size;
    }

    // policy state and initial fill: per layer the n_slot most frequent experts (ties -> lower id)
    const size_t ne = (size_t) n_layer*n_expert;
    map.assign(ne, -1);
    slot_of.assign(ne, -1);
    slot_expert.assign((size_t) n_layer*n_slot, -1);
    ema.assign(ne, 0.0);
    tie.resize(ne);
    for (size_t i = 0; i < ne; ++i) {
        tie[i] = (1e-6*(double) freq[i])/(double) fmax;
    }
    const auto t_fill = moe_clock::now();
    std::vector<int> order(n_expert);
    for (int il = 0; il < n_layer; ++il) {
        std::iota(order.begin(), order.end(), 0);
        const int64_t * fl = freq.data() + (size_t) il*n_expert;
        std::stable_sort(order.begin(), order.end(), [fl](int a, int b) { return fl[a] > fl[b]; });
        for (int s = 0; s < n_slot; ++s) {
            const int e = order[s];
            map[(size_t) il*n_expert + e]     = s;
            slot_of[(size_t) il*n_expert + e] = s;
            slot_expert[(size_t) il*n_slot + s] = e;
            queue_expert(il, e, s);
        }
        submit_dma();
    }
    ggml_backend_cuda_aux_synchronize(cuda);
    const double fill_ms = moe_hybrid_us(t_fill)/1e3;

    ev = ggml_backend_event_new(ggml_backend_get_device(cuda));
    if (ev == nullptr) {
        GGML_ABORT("moe_hybrid: event creation failed");
    }

    if (const char * path = getenv("LLAMA_MOE_HYBRID_LOG"); path != nullptr && *path != '\0') {
        log = fopen(path, "w");
        if (log == nullptr) {
            GGML_ABORT("moe_hybrid: cannot open log %s", path);
        }
        fprintf(log, "{\"event\":\"config\",\"slots\":%d,\"B\":%d,\"lambda\":%.17g,\"threads\":%d,\"first\":%d,\"stride\":%d,\"poll\":%d,"
                     "\"n_layer\":%d,\"n_expert\":%d,\"n_used\":%d,\"cache_bytes\":%zu,\"reg_bytes\":%zu,\"reg_ms\":%.1f,\"fill_ms\":%.1f}\n",
                n_slot, budget, lambda, n_threads, cpu_first, cpu_stride, poll, n_layer, n_expert, n_used,
                ggml_backend_buffer_get_size(cache_buf), reg_bytes, reg_ms, fill_ms);
        fflush(log);
    }

    // host miss engine
    std::promise<std::string> ready;
    std::future<std::string>  fut = ready.get_future();
    th = std::thread([this](std::promise<std::string> p) { run(std::move(p)); }, std::move(ready));
    const std::string err = fut.get();
    if (!err.empty()) {
        th.join();
        GGML_ABORT("moe_hybrid: host engine init failed: %s", err.c_str());
    }

    LLAMA_LOG_INFO("%s: moe_hybrid: %d slots x %d layers (%.1f MiB VRAM), B=%d lambda=%g, engine %d threads from CPU %d stride %d poll %d, "
                   "registered %.1f MiB in %.1f ms, fill %.1f ms, init %.1f ms\n", __func__,
                   n_slot, n_layer, ggml_backend_buffer_get_size(cache_buf)/1048576.0, budget, lambda, n_threads, cpu_first, cpu_stride, poll,
                   reg_bytes/1048576.0, reg_ms, fill_ms, moe_hybrid_us(t_init)/1e3);
}

std::string moe_hybrid_engine::build_host_graph(moe_hybrid_host_graph & hg, int l, int m) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false), nullptr, true };
    hg.ctx = ggml_init(ip);
    if (hg.ctx == nullptr) {
        return "graph context allocation failed";
    }
    ggml_context * ctx = hg.ctx;
    hg.x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    hg.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m, 1);
    hg.w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, m, 1);
    ggml_set_input(hg.x);
    ggml_set_input(hg.ids);
    ggml_set_input(hg.w);

    // same op sequence as the stock separate gate/up SiLU path of build_moe_ffn
    ggml_tensor * cur  = ggml_reshape_3d(ctx, hg.x, n_embd, 1, 1);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, h_up[l],   cur, hg.ids); // [n_ff, m, 1]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, h_gate[l], cur, hg.ids); // [n_ff, m, 1]
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * ex   = ggml_mul_mat_id(ctx, h_down[l], act, hg.ids); // [n_embd, m, 1]
    ex = ggml_mul(ctx, ex, hg.w);

    hg.gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(hg.gf, ex);
    ggml_tensor * views[GGML_MOE_HYBRID_MAX_USED];
    for (int i = 0; i < m; ++i) {
        views[i] = ggml_view_2d(ctx, ex, n_embd, 1, ex->nb[2], i*ex->nb[1]);
        ggml_build_forward_expand(hg.gf, views[i]);
    }
    ggml_tensor * out = views[0];
    for (int i = 1; i < m; ++i) {
        out = ggml_add(ctx, out, views[i]);
        ggml_build_forward_expand(hg.gf, out);
    }
    if (m == 1) {
        out = ggml_cont(ctx, out);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(hg.gf, out);
    hg.out = out;

    hg.ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (hg.ga == nullptr || !ggml_gallocr_alloc_graph(hg.ga, hg.gf)) {
        return "graph allocation failed";
    }
    return "";
}

void moe_hybrid_engine::run(std::promise<std::string> ready) {
    std::string err;

    ggml_cpu_init();
    cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        err = "CPU backend init failed";
    }
    if (err.empty()) {
        // created on this thread: in non-OpenMP builds the caller becomes worker 0 with the first CPU and the priority
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
        for (int i = 0; i < n_threads; ++i) {
            tpp.cpumask[cpu_first + i*cpu_stride] = true;
        }
        tpp.strict_cpu = true;
        tpp.poll       = (uint32_t) poll;
        tpp.prio       = GGML_SCHED_PRIO_HIGH;
        tp = ggml_threadpool_new(&tpp);
        if (tp == nullptr) {
            err = "threadpool creation failed";
        }
    }
    if (err.empty()) {
        ggml_backend_cpu_set_threadpool(cpu, tp);
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        graphs.resize((size_t) n_layer*n_used);
        for (int l = 0; l < n_layer && err.empty(); ++l) {
            for (int m = 1; m <= n_used && err.empty(); ++m) {
                err = build_host_graph(graphs[(size_t) l*n_used + m - 1], l, m);
            }
        }
    }

    const bool ok = err.empty();
    ready.set_value(err);
    if (ok) {
        serve();
    }

    for (auto & hg : graphs) {
        if (hg.ga) {
            ggml_gallocr_free(hg.ga);
        }
        if (hg.ctx) {
            ggml_free(hg.ctx);
        }
    }
    graphs.clear();
    if (cpu) {
        ggml_backend_cpu_set_threadpool(cpu, nullptr);
    }
    if (tp) {
        ggml_threadpool_free(tp);
        tp = nullptr;
    }
    if (cpu) {
        ggml_backend_free(cpu);
        cpu = nullptr;
    }
}

void moe_hybrid_engine::serve() {
    int32_t  served   = 0;
    uint32_t seen_gen = wake_gen.load(std::memory_order_acquire);
    uint32_t spins    = 0;
    auto     active   = moe_clock::now();

    while (!stop.load(std::memory_order_relaxed)) {
        const int32_t seq = mb->req_seq;
        if (seq == served) {
            MOE_HYBRID_PAUSE();
            if ((++spins & 4095) == 0) {
                const uint32_t gen = wake_gen.load(std::memory_order_acquire);
                if (gen != seen_gen) {
                    seen_gen = gen;
                    active   = moe_clock::now();
                } else if (moe_clock::now() - active > std::chrono::milliseconds(50)) {
                    // idle between generations: sleep until the next token wakes us (or 10 ms)
                    std::unique_lock<std::mutex> lk(mtx);
                    cv.wait_for(lk, std::chrono::milliseconds(10), [&] {
                        return stop.load() || wake_gen.load() != seen_gen;
                    });
                }
            }
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t0 = moe_clock::now();

        const int il    = mb->layer;
        const int nmiss = mb->nmiss;
        int32_t   err   = 0;
        if (il < 0 || il >= n_layer || nmiss < 1 || nmiss > n_used) {
            err = 1;
        } else {
            for (int i = 0; i < nmiss; ++i) {
                if (mb->ids[i] < 0 || mb->ids[i] >= n_expert) {
                    err = 1;
                }
            }
        }
        if (err == 0) {
            moe_hybrid_host_graph & hg = graphs[(size_t) il*n_used + nmiss - 1];
            memcpy(hg.x->data,   mb->x,   (size_t) n_embd*sizeof(float));
            memcpy(hg.ids->data, mb->ids, (size_t) nmiss*sizeof(int32_t));
            memcpy(hg.w->data,   mb->w,   (size_t) nmiss*sizeof(float));
            const auto        tc = moe_clock::now();
            const ggml_status st = ggml_backend_graph_compute(cpu, hg.gf);
            compute_sum[nmiss] += moe_hybrid_us(tc);
            if (st != GGML_STATUS_SUCCESS) {
                err = 2;
            } else {
                memcpy(mb->y, hg.out->data, (size_t) n_embd*sizeof(float));
            }
        }
        if (il >= 0 && il < n_layer) {
            mb->prof[il][3] = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(moe_clock::now() - t0).count();
        }
        mb->err = err;
        std::atomic_thread_fence(std::memory_order_release);
        mb->res_seq = seq;
        served = seq;

        const double us = moe_hybrid_us(t0);
        n_requests++;
        serve_sum += us;
        serve_max  = std::max(serve_max, us);
        nmiss_hist[err == 0 ? nmiss : 0]++;
        active = moe_clock::now();

        // the token's inputs are uploaded by now: start the promotion copies queued by on_token
        int queued = 1;
        if (dma_state.load(std::memory_order_relaxed) == 1 &&
            dma_state.compare_exchange_strong(queued, 2, std::memory_order_acquire)) {
            submit_dma();
            ggml_backend_cuda_event_record_aux(cuda, ev);
            ggml_backend_cuda_aux_flush(cuda);
            n_dma_engine++;
            dma_submit_sum += moe_hybrid_us(active);
            dma_state.store(0, std::memory_order_release);
            active = moe_clock::now();
        }
    }
}

void moe_hybrid_engine::on_token() {
    wake();
    const auto t0 = moe_clock::now();

    // the previous token's graph (and every host reply it waited for) is complete after this
    ggml_backend_synchronize(cuda);

    // copies the engine did not take (no host request since they were queued) are submitted here
    for (int st = dma_state.load(std::memory_order_acquire); st != 0; st = dma_state.load(std::memory_order_acquire)) {
        int queued = 1;
        if (st == 1 && dma_state.compare_exchange_strong(queued, 2, std::memory_order_acquire)) {
            submit_dma();
            ggml_backend_cuda_event_record_aux(cuda, ev);
            ggml_backend_cuda_aux_flush(cuda);
            n_dma_main++;
            dma_state.store(0, std::memory_order_release);
            break;
        }
        MOE_HYBRID_PAUSE();
    }

    // timeline of that graph in us (globaltimer ns in the mailbox, diagnostics only)
    double span = 0.0, stall = 0.0, hserve = 0.0, rt = 0.0, attn = 0.0, ovl = 0.0, ydl = 0.0, gap = -1.0;
    int    ml   = 0;
    bool   prof = false;
    {
        const uint64_t (* P)[5] = mb->prof;
        int first = -1;
        int prev  = -1;
        for (int l = 0; l < n_layer; ++l) {
            if (P[l][0] == 0 || P[l][2] == 0) {
                continue;
            }
            if (first < 0) {
                first = l;
            } else {
                attn += (double) (int64_t) (P[l][0] - P[prev][2]);
            }
            ovl   += (double) (int64_t) (P[l][1] - P[l][0]);
            stall += (double) (int64_t) (P[l][2] - P[l][1]);
            if (P[l][3] != 0) {
                hserve += (double) P[l][3];
                rt     += (double) (int64_t) (P[l][2] - P[l][0]);
                if (P[l][4] >= P[l][2]) {
                    ydl += (double) (int64_t) (P[l][4] - P[l][2]);
                }
                ml++;
            }
            prev = l;
        }
        if (first >= 0) {
            prof = true;
            span = (double) (int64_t) (P[prev][2] - P[first][0]);
            if (prev_end != 0) {
                gap = (double) (int64_t) (P[first][0] - prev_end)*1e-3;
            }
            prev_end = P[prev][2];
            span *= 1e-3; stall *= 1e-3; hserve *= 1e-3; rt *= 1e-3; attn *= 1e-3; ovl *= 1e-3; ydl *= 1e-3;
            memset(mb->prof, 0, sizeof(mb->prof));
        }
    }
    int  hits    = 0;
    int  lookups = 0;
    bool have    = false;
    const double decay = 1.0 - lambda;
    for (int l = 0; l < n_layer; ++l) {
        const int32_t * tr = trace + (size_t) l*GGML_MOE_HYBRID_MAX_USED;
        if (tr[0] < 0) {
            continue;
        }
        if (!have) {
            for (double & v : ema) {
                v *= lambda;
            }
            have = true;
        }
        for (int k = 0; k < n_used; ++k) {
            const int e = tr[k];
            if (e < 0 || e >= n_expert) {
                GGML_ABORT("moe_hybrid: corrupt routing trace (layer %d, entry %d = %d)", l, k, e);
            }
            const size_t i = (size_t) l*n_expert + e;
            lookups++;
            hits += map[i] >= 0;
            ema[i] += decay;
        }
    }
    std::fill(trace, trace + (size_t) n_layer*GGML_MOE_HYBRID_MAX_USED, -1);

    int  n_prom = 0;
    bool waited = false;
    if (!pending.empty()) {
        if (!ggml_backend_cuda_event_ready(ev)) {
            waited = true;
        } else {
            for (const moe_hybrid_promo & p : pending) {
                map[(size_t) p.l*n_expert + p.e] = p.s;
            }
            pending.clear();
        }
    }

    if (have && !waited && budget != 0) {
        pairs.clear();
        for (int l = 0; l < n_layer; ++l) {
            double cs[MOE_HYBRID_K]; int ce[MOE_HYBRID_K]; int nc = 0; // best uncached, score desc
            double vs[MOE_HYBRID_K]; int ve[MOE_HYBRID_K]; int nv = 0; // worst cached, score asc
            const size_t base = (size_t) l*n_expert;
            for (int e = 0; e < n_expert; ++e) {
                const double s = ema[base + e] + tie[base + e];
                if (slot_of[base + e] < 0) {
                    int p;
                    if (nc < MOE_HYBRID_K) {
                        p = nc++;
                    } else if (s > cs[MOE_HYBRID_K - 1]) {
                        p = MOE_HYBRID_K - 1;
                    } else {
                        continue;
                    }
                    while (p > 0 && cs[p - 1] < s) {
                        cs[p] = cs[p - 1];
                        ce[p] = ce[p - 1];
                        --p;
                    }
                    cs[p] = s;
                    ce[p] = e;
                } else {
                    int p;
                    if (nv < MOE_HYBRID_K) {
                        p = nv++;
                    } else if (s < vs[MOE_HYBRID_K - 1]) {
                        p = MOE_HYBRID_K - 1;
                    } else {
                        continue;
                    }
                    while (p > 0 && vs[p - 1] > s) {
                        vs[p] = vs[p - 1];
                        ve[p] = ve[p - 1];
                        --p;
                    }
                    vs[p] = s;
                    ve[p] = e;
                }
            }
            const int n = std::min(nc, nv);
            for (int k = 0; k < n; ++k) {
                const double gain = cs[k] - vs[k];
                if (!(gain > 0.0)) {
                    break;
                }
                pairs.push_back({ gain, l, ce[k], ve[k] });
            }
        }
        if (budget > 0 && (int) pairs.size() > budget) {
            std::stable_sort(pairs.begin(), pairs.end(), [](const moe_hybrid_pair & a, const moe_hybrid_pair & b) {
                return a.gain > b.gain;
            });
            pairs.resize(budget);
        }
        for (const moe_hybrid_pair & c : pairs) {
            const size_t iv = (size_t) c.l*n_expert + c.victim;
            const size_t ic = (size_t) c.l*n_expert + c.cand;
            const int    s  = slot_of[iv];
            GGML_ASSERT(s >= 0 && map[iv] == s && slot_of[ic] < 0);
            map[iv]     = -1;
            slot_of[iv] = -1;
            slot_of[ic] = s;
            slot_expert[(size_t) c.l*n_slot + s] = c.cand;
            pending.push_back({ c.l, c.cand, s });
            queue_expert(c.l, c.cand, s);
        }
        if (!pairs.empty()) {
            dma_state.store(1, std::memory_order_release);
        }
        n_prom = (int) pairs.size();
    }

    const double us = moe_hybrid_us(t0);
    if (prof) {
        n_prof++;
        n_ml       += ml;
        span_sum   += span;
        stall_sum  += stall;
        hserve_sum += hserve;
        rt_sum     += rt;
        ydl_sum    += ydl;
        attn_sum   += attn;
        ovl_sum    += ovl;
        // a longer gap spans a prompt, a warmup or a run boundary rather than one token step
        if (gap >= 0.0 && gap < 50000.0) {
            n_gap++;
            gap_sum += gap;
        }
    }
    if (have) {
        n_tokens++;
        n_hits       += hits;
        n_lookups    += lookups;
        n_promotions += n_prom;
        n_waits      += waited;
        on_token_sum += us;
        on_token_max  = std::max(on_token_max, us);
        if (log) {
            fprintf(log, "{\"t\":%lld,\"hits\":%d,\"lookups\":%d,\"prom\":%d,\"wait\":%d,\"us\":%.1f,"
                         "\"span\":%.1f,\"attn\":%.1f,\"ovl\":%.1f,\"stall\":%.1f,\"hserve\":%.1f,\"rt\":%.1f,\"ydl\":%.1f,\"ml\":%d,\"gap\":%.1f}\n",
                    (long long) n_tokens - 1, hits, lookups, n_prom, (int) waited, us,
                    span, attn, ovl, stall, hserve, rt, ydl, ml, gap);
        }
    }
}

void moe_hybrid_engine::shutdown() {
    if (down) {
        return;
    }
    down = true;
    if (th.joinable()) {
        stop.store(true);
        wake();
        th.join();
    }
    // copies still queued would only feed promotions that are never published
    dma_dst.clear();
    dma_src.clear();
    dma_size.clear();
    dma_state.store(0);
    if (cuda) {
        ggml_backend_cuda_aux_synchronize(cuda);
    }
    if (n_tokens > 0 || n_requests > 0) {
        char hist[256];
        int  off = 0;
        for (int m = 0; m <= n_used && off < (int) sizeof(hist); ++m) {
            off += snprintf(hist + off, sizeof(hist) - off, "%s%lld", m ? "," : "", (long long) nmiss_hist[m]);
        }
        LLAMA_LOG_INFO("moe_hybrid: tokens %lld, hit rate %.4f (%lld/%lld), promotions/token %.2f, pending waits %lld, "
                       "on_token avg %.1f max %.1f us | host requests %lld, serve avg %.1f max %.1f us, nmiss hist [err,1..%d] %s | "
                       "promotion batches engine %lld (submit avg %.1f us) main %lld\n",
                       (long long) n_tokens, n_lookups ? (double) n_hits/n_lookups : 0.0, (long long) n_hits, (long long) n_lookups,
                       n_tokens ? (double) n_promotions/n_tokens : 0.0, (long long) n_waits,
                       n_tokens ? on_token_sum/n_tokens : 0.0, on_token_max,
                       (long long) n_requests, n_requests ? serve_sum/n_requests : 0.0, serve_max, n_used, hist,
                       (long long) n_dma_engine, n_dma_engine ? dma_submit_sum/n_dma_engine : 0.0, (long long) n_dma_main);
        if (log) {
            fprintf(log, "{\"event\":\"summary\",\"tokens\":%lld,\"hits\":%lld,\"lookups\":%lld,\"promotions\":%lld,\"waits\":%lld,"
                         "\"on_token_avg_us\":%.2f,\"on_token_max_us\":%.1f,\"requests\":%lld,\"serve_avg_us\":%.2f,\"serve_max_us\":%.1f,"
                         "\"nmiss_hist\":[%s],\"dma_engine\":%lld,\"dma_submit_avg_us\":%.2f,\"dma_main\":%lld}\n",
                    (long long) n_tokens, (long long) n_hits, (long long) n_lookups, (long long) n_promotions, (long long) n_waits,
                    n_tokens ? on_token_sum/n_tokens : 0.0, on_token_max, (long long) n_requests,
                    n_requests ? serve_sum/n_requests : 0.0, serve_max, hist,
                    (long long) n_dma_engine, n_dma_engine ? dma_submit_sum/n_dma_engine : 0.0, (long long) n_dma_main);
        }
        char cmp[256];
        off = 0;
        for (int m = 1; m <= n_used && off < (int) sizeof(cmp); ++m) {
            off += snprintf(cmp + off, sizeof(cmp) - off, "%s%.1f", m > 1 ? "," : "",
                            nmiss_hist[m] ? compute_sum[m]/nmiss_hist[m] : 0.0);
        }
        const double np = n_prof ? (double) n_prof : 1.0;
        LLAMA_LOG_INFO("moe_hybrid: timeline per token over %lld tokens (us): span %.1f = attn %.1f + ovl %.1f + stall %.1f, "
                       "gap %.1f (%lld) | miss layers %.2f, host serve %.1f, publish->result %.1f, result load %.1f | cpu graph avg by miss count [1..%d] %s\n",
                       (long long) n_prof, span_sum/np, attn_sum/np, ovl_sum/np, stall_sum/np,
                       n_gap ? gap_sum/n_gap : -1.0, (long long) n_gap, n_ml/np, hserve_sum/np, rt_sum/np, ydl_sum/np, n_used, cmp);
        if (log) {
            fprintf(log, "{\"event\":\"timeline\",\"tokens\":%lld,\"span_us\":%.2f,\"attn_us\":%.2f,\"ovl_us\":%.2f,\"stall_us\":%.2f,"
                         "\"gap_us\":%.2f,\"gaps\":%lld,\"miss_layers\":%.3f,\"hserve_us\":%.2f,\"rt_us\":%.2f,\"ydl_us\":%.2f,\"cpu_graph_us\":[%s]}\n",
                    (long long) n_prof, span_sum/np, attn_sum/np, ovl_sum/np, stall_sum/np,
                    n_gap ? gap_sum/n_gap : -1.0, (long long) n_gap, n_ml/np, hserve_sum/np, rt_sum/np, ydl_sum/np, cmp);
        }
    }
    if (log) {
        fclose(log);
        log = nullptr;
    }
    if (ev) {
        ggml_backend_event_free(ev);
        ev = nullptr;
    }
    if (cache_buf) {
        ggml_backend_buffer_free(cache_buf);
        cache_buf = nullptr;
    }
    if (cache_ctx) {
        ggml_free(cache_ctx);
        cache_ctx = nullptr;
    }
    if (host_buf) {
        ggml_backend_buffer_free(host_buf);
        host_buf = nullptr;
    }
    for (const void * p : registered) {
        ggml_backend_cuda_unregister_host_range(p);
    }
    registered.clear();
    cuda = nullptr;
}

static std::mutex                                                     g_moe_hybrid_mtx;
static std::map<ggml_backend_t, std::unique_ptr<moe_hybrid_engine>> g_moe_hybrid_engines;

static moe_hybrid_engine * moe_hybrid_get(const llm_graph_context & g, const llama_model & model, int n_slot) {
    ggml_backend_t cuda   = nullptr;
    int            n_cuda = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(g.sched); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(g.sched, i);
        if (ggml_backend_is_cuda(b)) {
            cuda = b;
            n_cuda++;
        }
    }
    if (n_cuda != 1) {
        GGML_ABORT("moe_hybrid: needs exactly one CUDA backend, the scheduler has %d", n_cuda);
    }

    std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
    auto it = g_moe_hybrid_engines.find(cuda);
    if (it != g_moe_hybrid_engines.end()) {
        if (it->second->model != &model || it->second->n_slot != n_slot) {
            GGML_ABORT("moe_hybrid: engine of this backend was created for another model or slot count");
        }
        return it->second.get();
    }
    auto eng = std::make_unique<moe_hybrid_engine>();
    eng->init(g, model, cuda, n_slot);
    return (g_moe_hybrid_engines[cuda] = std::move(eng)).get();
}

// published slot map, refreshed (and the policy stepped) once per decoded token
class llm_graph_input_moe_hybrid : public llm_graph_input_i {
public:
    explicit llm_graph_input_moe_hybrid(moe_hybrid_engine * engine) : eng(engine) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        eng->on_token();
        ggml_backend_tensor_set(map, eng->map.data(), 0, ggml_nbytes(map));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.n_tokens == 1;
    }

    moe_hybrid_engine * eng;
    ggml_tensor *       map = nullptr; // I32 [n_layer*n_expert]
};

} // namespace

ggml_tensor * llama_moe_hybrid_build_ffn(const llm_graph_context & g, const llama_model & model, int il,
                                         ggml_tensor * cur, ggml_tensor ** pub) {
    *pub = nullptr;

    const int n_slot = moe_hybrid_env_int("LLAMA_MOE_HYBRID", 0);
    if (n_slot <= 0 || g.n_tokens != 1 || cur->ne[1] != 1 || g.layer_streaming != nullptr || il < 0 || il >= g.n_layer ||
        g.n_expert_used != (int64_t) g.hparams.n_expert_used || model.hparams.no_alloc) {
        return nullptr;
    }

    moe_hybrid_engine * eng = moe_hybrid_get(g, model, n_slot);

    llm_graph_input_moe_hybrid * inp = nullptr;
    for (auto & p : g.res->inputs) {
        if (auto * cand = dynamic_cast<llm_graph_input_moe_hybrid *>(p.get())) {
            inp = cand;
            break;
        }
    }
    if (inp == nullptr) {
        auto owned = std::make_unique<llm_graph_input_moe_hybrid>(eng);
        owned->map = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, (int64_t) eng->n_layer*eng->n_expert);
        ggml_set_input(owned->map);
        ggml_set_name(owned->map, "moe_hybrid_map");
        inp = static_cast<llm_graph_input_moe_hybrid *>(g.res->add_input(std::move(owned)));
    }

    ggml_context * ctx0 = g.ctx0;
    const llama_layer & L = model.layers[il];
    const int64_t n_embd        = cur->ne[0];
    const int64_t n_expert      = g.n_expert;
    const int64_t n_expert_used = g.n_expert_used;

    // router: identical to build_moe_ffn (softmax gating, norm_w, expert_weights_scale), so the topk-moe fusion still applies
    ggml_tensor * logits = g.build_lora_mm(L.ffn_gate_inp, cur); // [n_expert, 1]
    g.cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    g.cb(probs, "ffn_moe_probs", il);

    ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, n_expert_used); // [n_expert_used, 1]
    g.cb(selected->src[0], "ffn_moe_argsort", il);
    g.cb(selected, "ffn_moe_topk", il);

    probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, 1);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected); // [1, n_expert_used, 1]
    g.cb(weights, "ffn_moe_weights", il);

    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, 1);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
    g.cb(weights_sum, "ffn_moe_weights_sum", il);
    weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
    g.cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
    weights = ggml_div(ctx0, weights, weights_sum);
    g.cb(weights, "ffn_moe_weights_norm", il);
    weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, 1);

    const float w_scale = g.hparams.expert_weights_scale;
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        g.cb(weights, "ffn_moe_weights_scaled", il);
    }
    ggml_build_forward_expand(g.gf, weights);

    // cache slots for the selected experts (miss -> zero slot) and the host request for the misses
    ggml_tensor * p = ggml_moe_hybrid_publish(ctx0, selected, weights, cur, inp->map, eng->state, il,
                                              (int32_t) n_expert, eng->n_slot, eng->mb_dev, eng->trace_dev);
    g.cb(p, "ffn_moe_hybrid_pub", il);
    ggml_build_forward_expand(g.gf, p);

    ggml_tensor * ids  = ggml_view_2d(ctx0, p, n_expert_used, 1, n_expert_used*sizeof(int32_t), 0);
    ggml_tensor * cur3 = ggml_reshape_3d(ctx0, cur, n_embd, 1, 1);

    ggml_tensor * up = ggml_mul_mat_id(ctx0, eng->c_up[il], cur3, ids); // [n_ff, n_expert_used, 1]
    g.cb(up, "ffn_moe_up", il);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, eng->c_gate[il], cur3, ids);
    g.cb(gate, "ffn_moe_gate", il);
    ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
    g.cb(act, "ffn_moe_swiglu", il);
    ggml_tensor * experts = ggml_mul_mat_id(ctx0, eng->c_down[il], act, ids); // [n_embd, n_expert_used, 1]
    g.cb(experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);
    g.cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(g.gf, experts);

    ggml_tensor * views[GGML_MOE_HYBRID_MAX_USED];
    for (uint32_t i = 0; i < g.hparams.n_expert_used; ++i) {
        views[i] = ggml_view_2d(ctx0, experts, n_embd, 1, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(g.gf, views[i]);
    }
    ggml_tensor * moe_out = views[0];
    for (uint32_t i = 1; i < g.hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, views[i]);
        ggml_build_forward_expand(g.gf, moe_out);
    }
    if (g.hparams.n_expert_used == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }

    *pub = p;
    return moe_out;
}

ggml_tensor * llama_moe_hybrid_build_wait(const llm_graph_context & g, ggml_tensor * cur, ggml_tensor * pub) {
    ggml_tensor * out = ggml_moe_hybrid_wait_add(g.ctx0, cur, pub, pub->src[4]);
    g.cb(out, "ffn_moe_hybrid_out", ((const int32_t *) pub->op_params)[0]);
    return out;
}

void llama_moe_hybrid_release(ggml_backend_t backend) {
    std::unique_ptr<moe_hybrid_engine> eng;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        auto it = g_moe_hybrid_engines.find(backend);
        if (it == g_moe_hybrid_engines.end()) {
            return;
        }
        eng = std::move(it->second);
        g_moe_hybrid_engines.erase(it);
    }
    eng->shutdown();
}

#endif // LLAMA_LAYER_STREAMING
