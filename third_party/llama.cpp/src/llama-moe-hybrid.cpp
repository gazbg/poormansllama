#include "llama-moe-hybrid.h"

#ifdef LLAMA_LAYER_STREAMING

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-lean.h"

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
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

// defined in ggml/src/ggml-cuda/moe-hybrid.cu (not in ggml-cuda.h: every CUDA translation unit includes that header)
extern "C" {
#ifdef _WIN32
__declspec(dllimport)
#endif
void ggml_backend_cuda_moe_hybrid_set_timeout_ms(int ms);
#ifdef _WIN32
__declspec(dllimport)
#endif
int64_t ggml_backend_cuda_dbg_nodes_launched(void);
#ifdef _WIN32
__declspec(dllimport)
#endif
int ggml_backend_cuda_dbg_in_compute(void);
#ifdef _WIN32
__declspec(dllimport)
#endif
int ggml_backend_cuda_dbg_aux_idx(void);
#ifdef _WIN32
__declspec(dllimport)
#endif
void ggml_backend_cuda_moe_hybrid_set_xdump(void * dev_ptr, int n_tok);
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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

// The default size of the expert team. The workers busy-spin, so more of them than the physical cores the process may use makes them preempt each other and the decoding, engine and
// driver threads for a whole scheduler timeslice (2.4-3.7 ms in the stall log): on a 4-core CPU the flat default of 8 gave 6-8 tok/s where 4 threads gave 30-33. 8 is the size the team
// was tuned with (a machine with 8 or more physical cores keeps it). Linux only: elsewhere, or when the topology cannot be read, it stays 8.
static int moe_hybrid_default_threads() {
    int n = 8;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        std::set<int> cores;   // the lowest logical CPU of each sibling group identifies the physical core
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &allowed)) {
                continue;
            }
            const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list";
            FILE * f = fopen(path.c_str(), "r");
            int first = -1;
            const bool ok = f != nullptr && fscanf(f, "%d", &first) == 1 && first >= 0;
            if (f != nullptr) {
                fclose(f);
            }
            if (!ok) {
                cores.clear();   // no usable topology: keep the old default
                break;
            }
            cores.insert(first);
        }
        if (!cores.empty()) {
            n = std::min(n, (int) cores.size());
        }
    }
#endif // defined(__linux__)
    return n;
}

struct moe_hybrid_pf_abort_ctx {
    const volatile int32_t * req[GGML_MOE_HYBRID_MAX_WIDTH];
    const int32_t *          served;
};

// true when a mailbox holds a request that has not been answered: a prefetch in progress must give way to it
static bool moe_hybrid_pf_abort(void * p) {
    const moe_hybrid_pf_abort_ctx * c = (const moe_hybrid_pf_abort_ctx *) p;
    for (int k = 0; k < GGML_MOE_HYBRID_MAX_WIDTH; ++k) {
        if (*c->req[k] != c->served[k]) {
            return true;
        }
    }
    return false;
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

std::atomic<bool> g_moe_hybrid_keep_stats{false};   // see llama_moe_hybrid_set_keep_stats()

struct moe_hybrid_engine {
    const llama_model * model = nullptr;
    ggml_backend_t      cuda  = nullptr;

    int n_layer  = 0;
    int n_expert = 0;
    int n_used   = 0;
    int n_slot   = 0;
    int n_embd   = 0;
    // read by OTHER threads (the server's free-memory watcher) while resize() runs: they must never touch cache_buf, which resize() frees and re-creates
    std::atomic<int64_t> info_cache_bytes{0};
    std::atomic<int>     info_n_slot{0};      // 0 while the cache is being rebuilt

    int    budget     = 48;
    double lambda     = 0.97;
    int    n_threads  = 8;
    int    cpu_first  = 0;
    int    cpu_stride = 2;
    int    poll       = 100;
    int    pf_bytes   = 0; // LLAMA_MOE_HYBRID_PF: software-prefetch distance of the host mul_mat_id kernels in bytes (0 = off)
    int    timeout_ms = 0; // LLAMA_MOE_HYBRID_TIMEOUT_MS: device-side wait limit (0 = built-in 500 ms)
    FILE * trace_fp   = nullptr; // LLAMA_MOE_HYBRID_TRACE: per-token routing dump for offline policy replay
    int64_t dbg_stall_req = 0; // LLAMA_MOE_HYBRID_DEBUG_STALL=<request>:<ms>: test-only host stall at that request number
    int     dbg_stall_ms  = 0;

    // model host expert tensors per layer
    std::vector<ggml_tensor *> h_up, h_gate, h_down;
    std::vector<const void *>  registered;

    // G17: slots per layer. Uniform (n_slot everywhere) unless LLAMA_MOE_HYBRID_SPLIT names a file of n_layer weights (per-layer misses of a calibration replay); then layer l gets a share
    // proportional to weight^alpha, clipped to [split_min, split_max], with the same TOTAL n_slot*n_layer (so the VRAM is the same). `slot_off` indexes slot_expert.
    std::vector<int>              slots_l;      // [n_layer]
    std::vector<size_t>           slot_off;     // [n_layer + 1]
    std::string                   split_path;
    double                        split_alpha = 0.5;
    int                           split_min   = 24;
    int                           split_max   = 160;
    // G17: LLAMA_MOE_HYBRID_RECENCY=1: a promotion candidate must have been routed in the latest step (the cheap form of second-touch admission)
    bool                          recency = false;
    std::vector<uint8_t>          last_sel;     // [n_layer*n_expert], the experts routed in the latest ingested step (recency only)

    // VRAM cache per layer [ne0, ne1, slots_l[layer] + 1]; the last slot of a layer stays zero and absorbs the misses
    ggml_context *             cache_ctx = nullptr;
    ggml_backend_buffer_t      cache_buf = nullptr;
    std::vector<ggml_tensor *> c_up, c_gate, c_down;
    ggml_tensor *              state = nullptr; // I32[16]: [0] = publish sequence

    // pinned, device-mapped mailboxes (one per token of a step: a verify step of width W uses mbs[0 .. W)) + routing trace [token][n_layer][MAX_USED]
    ggml_backend_buffer_t     host_buf  = nullptr;
    ggml_moe_hybrid_mailbox * mb        = nullptr;     // = mbs[0]
    ggml_moe_hybrid_mailbox * mbs[GGML_MOE_HYBRID_MAX_WIDTH] = {};
    size_t                    mb_stride = 0;           // bytes between two mailboxes
    int32_t *                 trace     = nullptr;
    int                       trace_stride = 0;        // int32 entries per token row: n_layer*MAX_USED
    int                       prev_w    = 1;           // token rows the previous graph wrote into trace
    void *                    mb_dev    = nullptr;
    void *                    trace_dev = nullptr;

    // policy state
    std::vector<int32_t>          map;         // [n_layer*n_expert] published slot or -1 (graph input)
    std::vector<int32_t>          slot_of;     // [n_layer*n_expert] slot incl. pending, -1 if not resident
    std::vector<int32_t>          slot_expert; // [sum of slots_l], layer l at slot_off[l]
    std::vector<double>           ema;
    std::vector<double>           tie;
    std::vector<int64_t>          freq_;       // expert frequency prior (kept for resize)
    int64_t                       fmax_ = 0;
    std::vector<moe_hybrid_promo> pending;
    std::vector<moe_hybrid_pair>  pairs;

    // candidate scan placement (env LLAMA_MOE_HYBRID_SCAN, G11): 0 = inline in on_token, as it always was; 1 = "lag 1" inline, the deterministic reference of mode 2: the pairs
    // are scanned at the end of one on_token (against the state after its apply) and applied by the next, which has updated the EMA with one more token in between;
    // 2 = the same lag-1 pairs, scanned by a helper thread while the token's graph runs, so the scan (130-185 us) leaves the critical path between two tokens.
    // The helper reads ema / tie / slot_of only between a start_scan() and the join at the top of the next on_token, and the main thread writes them only after that join.
    int                           scan_mode = 0;
    // LLAMA_MOE_HYBRID_XDUMP (study tool): every layer's FFN input of the first xd_ntok decode steps, written to xd_path at shutdown
    ggml_backend_buffer_t         xd_buf  = nullptr;
    float *                       xd_host = nullptr;
    int                           xd_ntok = 0;
    std::string                   xd_path;
    int                           scan_mult = 1;        // token rows of the step the pairs will overlap: the promotion budget B is per token, so a verify step promotes up to B*scan_mult
    int                           scan_cpu  = 20;       // LLAMA_MOE_HYBRID_SCAN_CPU: logical CPU of the helper (-1 = unpinned); 20 is on the second CCD, away from the engine's CPUs 0,2,..,14
    std::vector<moe_hybrid_pair>  scan_out;
    bool                          scan_have = false;    // main thread only: scan_out holds (or will hold, after the join) a result not yet applied
    std::thread                   scan_th;
    std::mutex                    scan_mtx;
    std::condition_variable       scan_cv;
    bool                          scan_req  = false;    // guarded by scan_mtx
    bool                          scan_stop = false;    // guarded by scan_mtx
    std::atomic<int>              scan_busy{0};
    double                        ot_join   = 0.0;      // time on_token spent waiting for the helper (us, summed)
    std::vector<void *>           dma_dst;
    std::vector<const void *>     dma_src;
    std::vector<size_t>           dma_size;
    std::atomic<int>              dma_state{0}; // 0 idle, 1 copies queued by on_token, 2 being submitted
    ggml_backend_event_t          ev = nullptr;
    bool                          batch_ok = true;
    bool                          down     = false;

    // main-thread stats
    int64_t n_tokens      = 0;   // decode steps with a routing trace
    int64_t n_rows        = 0;   // token rows over those steps (a verify step has 1 + drafts rows)
    int64_t n_hits        = 0;
    int64_t n_lookups     = 0;
    int64_t n_promotions  = 0;
    int64_t n_waits       = 0;
    int64_t n_dma_main    = 0; // promotion batches the engine left to on_token (tokens without misses)
    double  on_token_sum  = 0.0;
    double  ot_sync = 0.0, ot_timeline = 0.0, ot_ema = 0.0, ot_pending = 0.0, ot_scan = 0.0, ot_apply = 0.0; // on_token section times (us, summed over the tokens)
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

    // default ON (LLAMA_MOE_HYBRID_DMA_THREAD=0 disables): the promotion copies are submitted by a thread of their own instead of by the request-serving engine thread, so a CUDA call
    // that blocks (the driver is shared with the main thread's kernel launches) can delay a promotion but never a reply the GPU is spinning on. dma_state 3 = handed over
    std::thread                        dma_th;
    std::mutex                         dma_mtx;
    std::condition_variable            dma_cv;
    bool                               dma_stop = false;     // guarded by dma_mtx
    bool                               dma_thread_on = false;

    // opt-in diagnostics (env LLAMA_MOE_HYBRID_WATCHDOG_MS > 0): a watchdog thread reports a host request that stays unanswered,
    // with the phase the host engine thread is in (0 polling for a request, 1 serving it, 2 submitting promotion copies)
    std::thread                        wd_th;
    std::atomic<bool>                  wd_stop{false};
    std::atomic<int>                   sv_phase{0};
    std::atomic<int64_t>               sv_t_ns{0};
    std::atomic<int32_t>               sv_seq{0};
    std::atomic<uint64_t>              sv_spins{0};  // heartbeat: +1 per 4096 polls of the request loop, so a thread that is not running stops counting
    std::atomic<int>                   sv_sub{0};    // inside a request: 11 copying inputs, 12 in the CPU graph compute, 13 copying the result
    int                                wd_ms = 0;
    double                             wd_max_ms = 0.0;  // worst pending time the watchdog observed (written by the watchdog thread only)
    int64_t                            wd_n_pend = 0;    // requests observed pending at two consecutive samples
    std::mutex                         mtx;
    std::condition_variable            cv;
    ggml_backend_t                     cpu = nullptr;
    ggml_threadpool *                  tp  = nullptr;
    std::vector<moe_hybrid_host_graph> graphs; // [n_layer*n_used], graph (l, m) serves m misses of layer l

    // G14 (env LLAMA_MOE_HYBRID_LEAN): the host leg that serves one request. 0 = the ggml graph above, as always (default); 1 = the lean team of llama-moe-lean.h: the same quantizers,
    // dot-product kernels and swiglu routine in ONE parallel region with three barriers instead of a graph of 8-12 nodes, so the result is bit-identical; 2 = both, compared bit for bit
    // on every request (a self-check: the graph's result is the one returned, every difference is counted and reported at shutdown). A model whose expert banks the lean team cannot
    // take keeps the graph. The engine thread is the team's thread 0.
    int                                lean_mode    = 0;
    int                                lean_spin_us = 4000;   // LLAMA_MOE_HYBRID_LEAN_SPIN_US: how long an idle worker spins before it parks
    int                                lean_prof    = 0;      // LLAMA_MOE_HYBRID_LEAN_PROF=1: stage times of the team (thread 0's view)
    std::string                        lean_cpus;             // LLAMA_MOE_HYBRID_LEAN_CPUS=a,b,c: logical CPUs of the team, the last one is the engine thread's (default: the engine's CPUs)
    int64_t                            lean_n_ff    = 0;
    std::vector<moe_lean::layer_banks> lean_lb;               // [n_layer]
    std::unique_ptr<moe_lean::team>    lean_team;
    std::vector<float>                 lean_y;                // mode 2: the lean result of the request
    moe_lean::stage_stats              lean_stats;
    int64_t                            n_lean = 0, n_lean_bad = 0;

    // G16 (env LLAMA_MOE_HYBRID_MAIN_CPU=n, Windows): the thread that decodes (it calls on_token between two tokens) is pinned to logical CPU n at the first decode step. Leaving core 0
    // (the machine's interrupt core) to the OS moved the decoding thread's wake-up after the logits copy onto it and made the gap between two tokens 230 us longer.
    int                                main_cpu     = -1;
    bool                               main_pinned  = false;

    // G15 (env LLAMA_MOE_HYBRID_LEAN_PF=K, needs the lean leg): after the reply to the request of layer l-1 the lean team predicts layer l's misses from the request's input row (a q8_0
    // copy of layer l's router, the first K experts of the top-POOL ranks that are not resident) and its workers prefetch those experts' row slices into their caches. No result can
    // change: a prefetch only touches memory. Counters below tell how good the prediction was in situ; the screen of tools/analyze-l2pf.py is their offline model.
    int                                lean_pf_k    = 0;
    int                                lean_pf_pool = 12;     // LLAMA_MOE_HYBRID_LEAN_PF_POOL
    int                                lean_pf_hint = 4;      // LLAMA_MOE_HYBRID_LEAN_PF_HINT: 0 prefetcht0, 1 prefetcht1 (L2), 2 prefetcht2, 3 prefetchnta, 4 plain loads
    std::vector<std::vector<char>>     pf_rw;                 // [n_layer] q8_0 router rows
    struct pf_rec {
        int32_t ids[moe_lean::MAX_PF] = {};
        int     n   = 0;
        int64_t tok = -1;                                      // decode step in which the post stage chose them
    };
    std::vector<pf_rec>                pf_last;               // [n_layer]
    int64_t                            pf_tok_no = 0;
    int64_t                            pf_trig = 0, pf_issued = 0, pf_useful = 0, pf_req_with_set = 0, pf_req_hit = 0, pf_miss_in_set_reqs = 0, pf_skipped = 0;
    double                             pf_post_sum = 0.0;
    int64_t                            n_requests = 0;
    double                             serve_sum  = 0.0;
    double                             serve_max  = 0.0;
    int64_t                            nmiss_hist[GGML_MOE_HYBRID_MAX_USED + 1] = {};
    double                             compute_sum[GGML_MOE_HYBRID_MAX_USED + 1] = {}; // CPU graph time by miss count
    int64_t                            n_dma_engine   = 0;
    double                             dma_submit_sum = 0.0;
    double                             dma_submit_max = 0.0;

    ~moe_hybrid_engine() { shutdown(); }

    void init(const llm_graph_context & g, const llama_model & mdl, ggml_backend_t be, int slots);
    void shutdown();
    void on_token(int n_new);
    bool resize(int new_slots);

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
    void        dma_loop();
    void        watchdog();
    void        compute_split();
    void        compute_pairs(std::vector<moe_hybrid_pair> & out) const;
    void        scan_loop();
    void        start_scan();
    void        join_scan();
    void        seed(const int32_t * ids, int K);
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
    n_threads  = moe_hybrid_env_int("LLAMA_MOE_HYBRID_THREADS", moe_hybrid_default_threads());
    if (const char * te = getenv("LLAMA_MOE_HYBRID_THREADS"); (te == nullptr || *te == '\0') && n_threads < 8) {
        LLAMA_LOG_WARN("%s: moe_hybrid: the expert team defaults to %d threads, the physical cores this process may use (LLAMA_MOE_HYBRID_THREADS overrides)\n", __func__, n_threads);
    }
    cpu_first  = moe_hybrid_env_int("LLAMA_MOE_HYBRID_FIRST", 0);
    cpu_stride = moe_hybrid_env_int("LLAMA_MOE_HYBRID_STRIDE", 2);
    poll       = moe_hybrid_env_int("LLAMA_MOE_HYBRID_POLL", 100);
    pf_bytes   = moe_hybrid_env_int("LLAMA_MOE_HYBRID_PF", 0);
    timeout_ms = moe_hybrid_env_int("LLAMA_MOE_HYBRID_TIMEOUT_MS", 0);
    if (const char * xp = getenv("LLAMA_MOE_HYBRID_XDUMP"); xp != nullptr && *xp != '\0') {
        xd_path = xp;
        xd_ntok = moe_hybrid_env_int("LLAMA_MOE_HYBRID_XDUMP_TOKENS", 512);
    }
    scan_mode  = moe_hybrid_env_int("LLAMA_MOE_HYBRID_SCAN", 0);
    scan_cpu   = moe_hybrid_env_int("LLAMA_MOE_HYBRID_SCAN_CPU", 20);
    if (scan_mode < 0 || scan_mode > 2) {
        GGML_ABORT("moe_hybrid: LLAMA_MOE_HYBRID_SCAN must be 0, 1 or 2 (got %d)", scan_mode);
    }
    lean_mode    = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN", 0);
    lean_spin_us = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_SPIN_US", 4000);
    lean_prof    = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_PROF", 0);
    if (const char * lc = getenv("LLAMA_MOE_HYBRID_LEAN_CPUS"); lc != nullptr && *lc != '\0') {
        lean_cpus = lc;
    }
    if (lean_mode < 0 || lean_mode > 2) {
        GGML_ABORT("moe_hybrid: LLAMA_MOE_HYBRID_LEAN must be 0, 1 or 2 (got %d)", lean_mode);
    }
    main_cpu     = moe_hybrid_env_int("LLAMA_MOE_HYBRID_MAIN_CPU", -2);    // -2 = automatic (see run()), -1 = never pin
    if (const char * sp = getenv("LLAMA_MOE_HYBRID_SPLIT"); sp != nullptr && *sp != '\0') {
        split_path = sp;
    }
    split_alpha  = moe_hybrid_env_double("LLAMA_MOE_HYBRID_SPLIT_ALPHA", 0.5);
    split_min    = moe_hybrid_env_int("LLAMA_MOE_HYBRID_SPLIT_MIN", 24);
    split_max    = moe_hybrid_env_int("LLAMA_MOE_HYBRID_SPLIT_MAX", 160);
    recency      = moe_hybrid_env_int("LLAMA_MOE_HYBRID_RECENCY", 0) != 0;
    lean_pf_k    = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_PF", 0);
    lean_pf_pool = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_PF_POOL", 12);
    lean_pf_hint = moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_PF_HINT", 4);
    if (lean_pf_k < 0 || lean_pf_k > moe_lean::MAX_PF || lean_pf_pool < 1 || lean_pf_pool > 16 || lean_pf_hint < 0 || lean_pf_hint > 4) {
        GGML_ABORT("moe_hybrid: bad LLAMA_MOE_HYBRID_LEAN_PF settings (K %d of at most %d, pool %d of 1..16, hint %d of 0..4)", lean_pf_k, moe_lean::MAX_PF, lean_pf_pool, lean_pf_hint);
    }
    if (lean_pf_k > 0 && lean_mode == 0) {
        LLAMA_LOG_WARN("%s: moe_hybrid: LLAMA_MOE_HYBRID_LEAN_PF needs LLAMA_MOE_HYBRID_LEAN >= 1; ignored\n", __func__);
        lean_pf_k = 0;
    }
    if (const char * tp_path = getenv("LLAMA_MOE_HYBRID_TRACE"); tp_path != nullptr && *tp_path != '\0') {
        trace_fp = fopen(tp_path, "wb");
        if (trace_fp == nullptr) {
            GGML_ABORT("moe_hybrid: cannot open routing trace %s", tp_path);
        }
        const int32_t hdr[4] = { (int32_t) g.n_layer, (int32_t) g.hparams.n_expert_used, (int32_t) g.n_expert, GGML_MOE_HYBRID_MAX_USED };
        fwrite("MOETRC01", 1, 8, trace_fp);
        fwrite(hdr, sizeof(int32_t), 4, trace_fp);
    }
#ifdef _WIN32
    if (const int prio = moe_hybrid_env_int("LLAMA_MOE_HYBRID_PRIO", 0); prio > 0) {
        const DWORD cls = prio >= 2 ? HIGH_PRIORITY_CLASS : ABOVE_NORMAL_PRIORITY_CLASS;
        if (SetPriorityClass(GetCurrentProcess(), cls)) {
            LLAMA_LOG_INFO("%s: moe_hybrid: process priority class raised (%s)\n", __func__, prio >= 2 ? "high" : "above normal");
        } else {
            LLAMA_LOG_WARN("%s: moe_hybrid: SetPriorityClass failed (%lu)\n", __func__, (unsigned long) GetLastError());
        }
    }
#endif
    ggml_backend_cuda_moe_hybrid_set_timeout_ms(timeout_ms);
    if (const char * ds = getenv("LLAMA_MOE_HYBRID_DEBUG_STALL"); ds != nullptr && *ds != '\0') {
        long long r = 0; int ms = 0;
        if (sscanf(ds, "%lld:%d", &r, &ms) == 2 && r > 0 && ms > 0) {
            dbg_stall_req = r;
            dbg_stall_ms  = ms;
            LLAMA_LOG_WARN("%s: moe_hybrid: DEBUG stall injection armed: host request %lld sleeps %d ms\n", __func__, r, ms);
        }
    }

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

    if (lean_mode > 0) {
        lean_lb.resize((size_t) n_layer);
        lean_n_ff = h_up[0]->ne[1];
        for (int il = 0; il < n_layer; ++il) {
            moe_lean::layer_banks & lb = lean_lb[(size_t) il];
            lb.up   = moe_lean::bank_of(h_up[il]);
            lb.gate = moe_lean::bank_of(h_gate[il]);
            lb.down = moe_lean::bank_of(h_down[il]);
            if (!moe_lean::supported(lb) || lb.up.k != n_embd || lb.down.rows != n_embd || lb.up.rows != lean_n_ff) {
                LLAMA_LOG_WARN("%s: moe_hybrid: layer %d has expert banks the lean host leg cannot take (types %s / %s / %s); LLAMA_MOE_HYBRID_LEAN is ignored and the graph serves every request\n",
                               __func__, il, ggml_type_name(lb.up.type), ggml_type_name(lb.gate.type), ggml_type_name(lb.down.type));
                lean_mode = 0;
                lean_lb.clear();
                lean_pf_k = 0;
                break;
            }
        }
    }
    if (lean_mode > 0 && lean_pf_k > 0) {
        if (n_embd % 32 != 0) {
            LLAMA_LOG_WARN("%s: moe_hybrid: n_embd %d is not a multiple of 32; the lean prefetch is off\n", __func__, n_embd);
            lean_pf_k = 0;
        } else {
            const auto t_rw = moe_clock::now();
            pf_rw.resize((size_t) n_layer);
            pf_last.assign((size_t) n_layer, pf_rec());
            std::vector<float> rows((size_t) n_expert*n_embd);
            for (int il = 0; il < n_layer && lean_pf_k > 0; ++il) {
                const ggml_tensor * t = mdl.layers[il].ffn_gate_inp;
                if (t == nullptr || t->ne[0] != n_embd || t->ne[1] != n_expert || t->ne[2] != 1 || t->ne[3] != 1) {
                    LLAMA_LOG_WARN("%s: moe_hybrid: layer %d has no [n_embd, n_expert] router tensor; the lean prefetch is off\n", __func__, il);
                    lean_pf_k = 0;
                    break;
                }
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(t, rows.data(), 0, ggml_nbytes(t));
                } else {
                    const ggml_type_traits * tt = ggml_get_type_traits(t->type);
                    if (tt == nullptr || tt->to_float == nullptr) {
                        LLAMA_LOG_WARN("%s: moe_hybrid: router type %s cannot be converted; the lean prefetch is off\n", __func__, ggml_type_name(t->type));
                        lean_pf_k = 0;
                        break;
                    }
                    std::vector<char> raw(ggml_nbytes(t));
                    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
                    for (int e = 0; e < n_expert; ++e) {
                        tt->to_float(raw.data() + (size_t) e*t->nb[1], rows.data() + (size_t) e*n_embd, n_embd);
                    }
                }
                pf_rw[(size_t) il] = moe_lean::quantize_router_q8(rows.data(), n_expert, n_embd);
            }
            if (lean_pf_k > 0) {
                LLAMA_LOG_INFO("%s: moe_hybrid: lean prefetch K=%d pool %d hint %d, q8_0 router copies of %d layers (%.1f MiB) in %.1f ms\n", __func__, lean_pf_k, lean_pf_pool, lean_pf_hint,
                               n_layer, (double) n_layer*pf_rw[0].size()/1048576.0, moe_hybrid_us(t_rw)/1e3);
            } else {
                pf_rw.clear();
                pf_last.clear();
            }
        }
    }

    // expert frequencies: initial fill and tie-break
    std::vector<int64_t> freq((size_t) n_layer*n_expert);
    {
        const char * path = getenv("LLAMA_MOE_HYBRID_FREQ");
        if (path == nullptr || *path == '\0') {
            // G31: a server need not ship a calibration table: without one the initial fill and the tie-break use a uniform prior
            std::fill(freq.begin(), freq.end(), (int64_t) 1);
            LLAMA_LOG_INFO("%s: moe_hybrid: no LLAMA_MOE_HYBRID_FREQ table, uniform expert prior\n", __func__);
        } else {
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
    }
    const int64_t fmax = *std::max_element(freq.begin(), freq.end());
    if (fmax <= 0 || *std::min_element(freq.begin(), freq.end()) < 0) {
        GGML_ABORT("moe_hybrid: invalid frequency counts");
    }
    freq_ = freq;
    fmax_ = fmax;

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
                if (strstr(err, "already mapped") != nullptr) {
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
    compute_split();
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
                ggml_tensor * t = ggml_new_tensor_3d(cache_ctx, src[k]->type, src[k]->ne[0], src[k]->ne[1], slots_l[(size_t) il] + 1);
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
        info_cache_bytes.store((int64_t) ggml_backend_buffer_get_size(cache_buf));
        info_n_slot.store(n_slot);
    }

    // mailboxes (one per token of a step) + trace
    {
        const size_t mb_size = GGML_PAD(sizeof(ggml_moe_hybrid_mailbox), 256);
        trace_stride = n_layer*GGML_MOE_HYBRID_MAX_USED;
        const size_t mb_all  = (size_t) GGML_MOE_HYBRID_MAX_WIDTH*mb_size;
        const size_t tr_size = (size_t) GGML_MOE_HYBRID_MAX_WIDTH*trace_stride*sizeof(int32_t);
        host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), mb_all + tr_size);
        if (host_buf == nullptr) {
            GGML_ABORT("moe_hybrid: pinned mailbox allocation failed");
        }
        char * base = (char *) ggml_backend_buffer_get_base(host_buf);
        memset(base, 0, mb_all + tr_size);
        mb_stride = mb_size;
        for (int j = 0; j < GGML_MOE_HYBRID_MAX_WIDTH; ++j) {
            mbs[j] = (ggml_moe_hybrid_mailbox *) (base + (size_t) j*mb_size);
        }
        mb    = mbs[0];
        trace = (int32_t *) (base + mb_all);
        std::fill(trace, trace + (size_t) GGML_MOE_HYBRID_MAX_WIDTH*trace_stride, -1);
        char * dev = (char *) ggml_backend_cuda_host_device_ptr(base);
        if (dev == nullptr) {
            GGML_ABORT("moe_hybrid: mailbox is not device-mapped pinned memory");
        }
        mb_dev    = dev;
        trace_dev = dev + mb_all;
    }

    // policy state and initial fill: per layer the n_slot most frequent experts (ties -> lower id)
    const size_t ne = (size_t) n_layer*n_expert;
    map.assign(ne, -1);
    slot_of.assign(ne, -1);
    slot_expert.assign(slot_off[(size_t) n_layer], -1);
    last_sel.assign(recency ? ne : 0, 0);
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
        for (int s = 0; s < slots_l[(size_t) il]; ++s) {
            const int e = order[s];
            map[(size_t) il*n_expert + e]     = s;
            slot_of[(size_t) il*n_expert + e] = s;
            slot_expert[slot_off[(size_t) il] + s] = e;
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

    if (xd_ntok > 0) {
        const size_t xd_bytes = (size_t) xd_ntok*n_layer*n_embd*sizeof(float);
        xd_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), xd_bytes);
        if (xd_buf == nullptr) {
            GGML_ABORT("moe_hybrid: cannot allocate the %.1f MiB FFN-input capture ring", xd_bytes/1048576.0);
        }
        xd_host = (float *) ggml_backend_buffer_get_base(xd_buf);
        memset(xd_host, 0, xd_bytes);
        void * xd_dev = ggml_backend_cuda_host_device_ptr(xd_host);
        if (xd_dev == nullptr) {
            GGML_ABORT("moe_hybrid: the capture ring is not device-mapped pinned memory");
        }
        ggml_backend_cuda_moe_hybrid_set_xdump(xd_dev, xd_ntok);
        LLAMA_LOG_INFO("%s: moe_hybrid: capturing the FFN input of %d decode steps x %d layers (%.1f MiB) for %s\n", __func__, xd_ntok, n_layer, xd_bytes/1048576.0, xd_path.c_str());
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

    dma_thread_on = moe_hybrid_env_int("LLAMA_MOE_HYBRID_DMA_THREAD", 1) != 0; // default ON since G5 (see docs/plans/tg-beat-stock-8gb-262k.md)
    if (dma_thread_on) {
        dma_th = std::thread([this]() { dma_loop(); });
    }
    if (scan_mode == 2 && budget != 0) {
        scan_th = std::thread([this]() { scan_loop(); });
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

    wd_ms = moe_hybrid_env_int("LLAMA_MOE_HYBRID_WATCHDOG_MS", 0);
    if (wd_ms > 0) {
        wd_th = std::thread([this]() { watchdog(); });
    }

    LLAMA_LOG_INFO("%s: moe_hybrid: %d slots x %d layers (%.1f MiB VRAM), B=%d lambda=%g, engine %d threads from CPU %d stride %d poll %d pf %d, "
                   "registered %.1f MiB in %.1f ms, fill %.1f ms, init %.1f ms\n", __func__,
                   n_slot, n_layer, ggml_backend_buffer_get_size(cache_buf)/1048576.0, budget, lambda, n_threads, cpu_first, cpu_stride, poll, pf_bytes,
                   reg_bytes/1048576.0, reg_ms, fill_ms, moe_hybrid_us(t_init)/1e3);
}

// Resize the VRAM cache between two decodes. The layer-streaming runner calls this right after prefill released its weight
// slots, so the decode phase can use that VRAM for a bigger cache. The caller guarantees that no hybrid graph is executing
// (host engine idle, nothing queued). The old cache is freed BEFORE the new one is allocated, so the two never coexist; the
// initial fill is the same frequency-ordered one init() does, and the publish sequence in `state` survives. Returns false
// and changes nothing when the request is invalid or the engine is not idle.
bool moe_hybrid_engine::resize(int new_slots) {
    if (new_slots == n_slot) {
        return true;
    }
    if (new_slots < 1 || new_slots > n_expert || freq_.empty() || cache_buf == nullptr) {
        return false;
    }
    // G31n: promotion copies of the last decoded token may still be queued (state 1): they target the cache that is about to be freed, drop them; wait (<= 100 ms) for a submission in progress (2, 3)
    {
        const auto tw = moe_clock::now();
        for (int st = dma_state.load(std::memory_order_acquire); st != 0; st = dma_state.load(std::memory_order_acquire)) {
            int queued = 1;
            if (st == 1 && dma_state.compare_exchange_strong(queued, 2, std::memory_order_acquire)) {
                dma_dst.clear();
                dma_src.clear();
                dma_size.clear();
                dma_state.store(0, std::memory_order_release);
                break;
            }
            if (moe_hybrid_us(tw) > 100000.0) {
                return false;
            }
            MOE_HYBRID_PAUSE();
        }
    }
    const auto t0 = moe_clock::now();
    info_n_slot.store(0);
    ggml_backend_cuda_aux_synchronize(cuda);
    join_scan();
    scan_have = false;
    scan_out.clear();

    int32_t st[16] = {};
    ggml_backend_tensor_get(state, st, 0, sizeof(st));

    ggml_backend_buffer_free(cache_buf);
    cache_buf = nullptr;
    ggml_free(cache_ctx);
    cache_ctx = nullptr;
    n_slot = new_slots;
    compute_split();

    ggml_init_params ip = { ggml_tensor_overhead()*(3*(size_t) n_layer + 1), nullptr, true };
    cache_ctx = ggml_init(ip);
    if (cache_ctx == nullptr) {
        GGML_ABORT("moe_hybrid: cache context allocation failed");
    }
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor ** dst[3] = { &c_up[il], &c_gate[il], &c_down[il] };
        const ggml_tensor * src[3] = { h_up[il], h_gate[il], h_down[il] };
        for (int k = 0; k < 3; ++k) {
            ggml_tensor * t = ggml_new_tensor_3d(cache_ctx, src[k]->type, src[k]->ne[0], src[k]->ne[1], slots_l[(size_t) il] + 1);
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
    ggml_backend_tensor_set(state, st, 0, sizeof(st));

    // policy state and initial fill, exactly as init() does for the new slot count. With keep_stats (the server, G31j/G31q) the popularity statistics (ema, last_sel) are KEPT and the refill takes the experts with the
    // highest policy score, so a shrink keeps the experts the engine found hot; without it (the runner) the statistics are reset and the order is the frequency table's, as recorded
    const bool   keep_stats = g_moe_hybrid_keep_stats.load();
    const size_t ne = (size_t) n_layer*n_expert;
    map.assign(ne, -1);
    slot_of.assign(ne, -1);
    slot_expert.assign(slot_off[(size_t) n_layer], -1);
    if (!keep_stats) {
        last_sel.assign(recency ? ne : 0, 0);
        ema.assign(ne, 0.0);
    }
    pending.clear();
    pairs.clear();
    dma_dst.clear();
    dma_src.clear();
    dma_size.clear();
    std::vector<int> order(n_expert);
    for (int il = 0; il < n_layer; ++il) {
        std::iota(order.begin(), order.end(), 0);
        if (keep_stats) {
            const double * em = ema.data() + (size_t) il*n_expert;
            const double * tt = tie.data() + (size_t) il*n_expert;
            std::stable_sort(order.begin(), order.end(), [em, tt](int a, int b) { return em[a] + tt[a] > em[b] + tt[b]; });
        } else {
            const int64_t * fl = freq_.data() + (size_t) il*n_expert;
            std::stable_sort(order.begin(), order.end(), [fl](int a, int b) { return fl[a] > fl[b]; });
        }
        for (int s = 0; s < slots_l[(size_t) il]; ++s) {
            const int e = order[s];
            map[(size_t) il*n_expert + e]     = s;
            slot_of[(size_t) il*n_expert + e] = s;
            slot_expert[slot_off[(size_t) il] + s] = e;
            queue_expert(il, e, s);
        }
        submit_dma();
    }
    ggml_backend_cuda_aux_synchronize(cuda);
    info_cache_bytes.store((int64_t) ggml_backend_buffer_get_size(cache_buf));
    info_n_slot.store(n_slot);
    LLAMA_LOG_INFO("%s: moe_hybrid: cache resized to %d slots x %d layers (%.1f MiB VRAM) in %.1f ms\n", __func__,
                   n_slot, n_layer, ggml_backend_buffer_get_size(cache_buf)/1048576.0, moe_hybrid_us(t0)/1e3);
    return true;
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
    ((int32_t *) up->op_params)[0]   = pf_bytes; // opt-in prefetch distance, read by ggml-cpu's mul_mat_id
    ((int32_t *) gate->op_params)[0] = pf_bytes;
    ((int32_t *) ex->op_params)[0]   = pf_bytes;
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
    const bool need_graph = lean_mode != 1; // mode 1 serves every request with the lean team: no ggml thread pool and no host graphs
    if (need_graph) {
        cpu = ggml_backend_cpu_init();
        if (cpu == nullptr) {
            err = "CPU backend init failed";
        }
    }
    if (err.empty() && need_graph) {
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
    if (err.empty() && need_graph) {
        ggml_backend_cpu_set_threadpool(cpu, tp);
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        graphs.resize((size_t) n_layer*n_used);
        for (int l = 0; l < n_layer && err.empty(); ++l) {
            for (int m = 1; m <= n_used && err.empty(); ++m) {
                err = build_host_graph(graphs[(size_t) l*n_used + m - 1], l, m);
            }
        }
    }
    if (err.empty() && lean_mode > 0) {
        std::vector<int> cpus;
        if (!lean_cpus.empty()) {
            size_t pos = 0;
            while (pos < lean_cpus.size()) {
                size_t nx = lean_cpus.find(',', pos);
                if (nx == std::string::npos) {
                    nx = lean_cpus.size();
                }
                cpus.push_back(atoi(lean_cpus.substr(pos, nx - pos).c_str()));
                pos = nx + 1;
            }
        } else {
            // core 0 (logical CPUs 0 and 1) takes the machine's interrupts and held the worker pinned to CPU 0 off for 0.6-100 ms in every stall the diagnostic ever reported (G16)
            const int first_eff = cpu_first == 0 ? 2 : cpu_first;
            for (int i = 0; i < n_threads; ++i) {
                cpus.push_back(first_eff + i*cpu_stride);
            }
        }
        if (main_cpu == -2) {
            // with the team off core 0 the scheduler put the decoding thread's wake-up after the logits copy onto it: the gap between two tokens grew by 230 us. Pin it away.
            main_cpu = (!cpus.empty() && cpus.back() + 2 < 64) ? cpus.back() + 2 : -1;
        }
        if (cpus.empty()) {
            err = "LLAMA_MOE_HYBRID_LEAN_CPUS names no CPU";
        } else {
            const auto pin = [](int cpu_id) {
#ifdef _WIN32
                if (cpu_id >= 0 && cpu_id < 64) {
                    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu_id);
                }
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
                (void) cpu_id;
#endif
            };
            lean_team = std::make_unique<moe_lean::team>((int) cpus.size(), n_embd, lean_n_ff, lean_pf_k > 0 ? n_expert : 0, [cpus, pin](int ith) { pin(cpus[(size_t) ith - 1]); }, lean_spin_us, lean_prof != 0);
            pin(cpus.back()); // the engine thread is thread 0 of the team
            lean_team->set_stall_logger((double) moe_hybrid_env_int("LLAMA_MOE_HYBRID_LEAN_STALL_US", 3000), [](const char * s) { LLAMA_LOG_WARN("%s\n", s); });
            lean_y.assign((size_t) n_embd, 0.0f);
            std::string list;
            for (int c : cpus) {
                list += (list.empty() ? "" : ",") + std::to_string(c);
            }
            LLAMA_LOG_INFO("%s: moe_hybrid: lean host leg mode %d, %zu threads on CPUs %s, idle spin %d us%s\n", __func__, lean_mode, cpus.size(), list.c_str(), lean_spin_us,
                           lean_mode == 2 ? " (SELF-CHECK: every request is also served by the graph and compared bit for bit)" : "");
        }
    }

    const bool ok = err.empty();
    ready.set_value(err);
    if (ok) {
        serve();
    }
    if (lean_team) {
        lean_stats = lean_team->stats();
        lean_team.reset();
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
    int32_t  served[GGML_MOE_HYBRID_MAX_WIDTH] = {};
    moe_hybrid_pf_abort_ctx pf_ctx = {};
    for (int k = 0; k < GGML_MOE_HYBRID_MAX_WIDTH; ++k) {
        pf_ctx.req[k] = &mbs[k]->req_seq;
    }
    pf_ctx.served = served;
    uint32_t seen_gen = wake_gen.load(std::memory_order_acquire);
    uint32_t spins    = 0;
    auto     active   = moe_clock::now();

    while (!stop.load(std::memory_order_relaxed)) {
        // the lowest-numbered mailbox with an unanswered request goes first: a verify step posts one request per token and layer, all published together by the one launch
        int j = -1;
        for (int k = 0; k < GGML_MOE_HYBRID_MAX_WIDTH; ++k) {
            if (mbs[k]->req_seq != served[k]) {
                j = k;
                break;
            }
        }
        if (j < 0) {
            MOE_HYBRID_PAUSE();
            if ((++spins & 4095) == 0) {
                sv_spins.fetch_add(1, std::memory_order_relaxed);
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
        ggml_moe_hybrid_mailbox * m = mbs[j];
        const int32_t seq = m->req_seq;
        const auto t0 = moe_clock::now();
        sv_seq.store(seq, std::memory_order_relaxed);
        sv_t_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count(), std::memory_order_relaxed);
        sv_phase.store(1, std::memory_order_relaxed);
        sv_sub.store(11, std::memory_order_relaxed);
        if (dbg_stall_ms > 0 && n_requests + 1 == dbg_stall_req) {
            std::this_thread::sleep_for(std::chrono::milliseconds(dbg_stall_ms)); // test-only
        }

        const int il    = m->layer;
        const int nmiss = m->nmiss;
        int32_t   err   = 0;
        double    in_us = 0.0, comp_us = 0.0;   // stage times of this request (diagnostics)
        if (il < 0 || il >= n_layer || nmiss < 1 || nmiss > n_used) {
            err = 1;
        } else {
            for (int i = 0; i < nmiss; ++i) {
                if (m->ids[i] < 0 || m->ids[i] >= n_expert) {
                    err = 1;
                }
            }
        }
        if (err == 0) {
            // the decode step: on_token (main thread, between two steps) counts them; a racy read of that counter is enough to tell a prediction of this step from a stale one
            pf_tok_no = *(volatile const int64_t *) &n_tokens;
            if (lean_pf_k > 0 && pf_last[(size_t) il].tok == pf_tok_no) {
                // the post stage of the previous request chose experts for THIS layer in this step: were they the ones that missed?
                const pf_rec & pr = pf_last[(size_t) il];
                int hits = 0;
                for (int i = 0; i < nmiss; ++i) {
                    for (int k = 0; k < pr.n; ++k) {
                        hits += m->ids[i] == pr.ids[k];
                    }
                }
                pf_req_with_set++;
                pf_useful += hits;
                pf_req_hit += hits > 0;
                pf_miss_in_set_reqs += nmiss;
                pf_last[(size_t) il].tok = -1;
            }
        }
        bool pf_job = false;
        if (err == 0 && lean_mode == 1) {
            // the lean team reads the request straight from the mailbox and writes the result straight into it
            moe_lean::job jb;
            jb.lb  = &lean_lb[(size_t) il];
            jb.m   = nmiss;
            jb.ids = (const int32_t *) m->ids;
            jb.w   = (const float *) m->w;
            jb.x   = (const float *) m->x;
            jb.y   = (float *) m->y;
            if (lean_pf_k > 0 && il + 1 < n_layer) {
                jb.pf_lb   = &lean_lb[(size_t) il + 1];
                jb.pf_rw   = pf_rw[(size_t) il + 1].data();
                jb.pf_map  = map.data() + (size_t) (il + 1)*n_expert;
                jb.pf_k    = lean_pf_k;
                jb.pf_pool = lean_pf_pool;
                jb.pf_hint = lean_pf_hint;
                jb.pf_abort = moe_hybrid_pf_abort;
                jb.pf_abort_ctx = &pf_ctx;
                pf_job     = true;
            }
            const auto tc = moe_clock::now();
            in_us = moe_hybrid_us(t0);
            sv_sub.store(12, std::memory_order_relaxed);
            lean_team->run(jb);
            sv_sub.store(13, std::memory_order_relaxed);
            comp_us = moe_hybrid_us(tc);
            compute_sum[nmiss] += comp_us;
            n_lean++;
        } else if (err == 0) {
            moe_hybrid_host_graph & hg = graphs[(size_t) il*n_used + nmiss - 1];
            memcpy(hg.x->data,   m->x,   (size_t) n_embd*sizeof(float));
            memcpy(hg.ids->data, m->ids, (size_t) nmiss*sizeof(int32_t));
            memcpy(hg.w->data,   m->w,   (size_t) nmiss*sizeof(float));
            const auto        tc = moe_clock::now();
            in_us = moe_hybrid_us(t0);
            sv_sub.store(12, std::memory_order_relaxed);
            const ggml_status st = ggml_backend_graph_compute(cpu, hg.gf);
            sv_sub.store(13, std::memory_order_relaxed);
            comp_us = moe_hybrid_us(tc);
            compute_sum[nmiss] += comp_us;
            if (st != GGML_STATUS_SUCCESS) {
                err = 2;
            } else {
                if (lean_mode == 2) {
                    moe_lean::job jb;
                    jb.lb  = &lean_lb[(size_t) il];
                    jb.m   = nmiss;
                    jb.ids = (const int32_t *) m->ids;
                    jb.w   = (const float *) m->w;
                    jb.x   = (const float *) m->x;
                    jb.y   = lean_y.data();
                    if (lean_pf_k > 0 && il + 1 < n_layer) {
                        jb.pf_lb   = &lean_lb[(size_t) il + 1];
                        jb.pf_rw   = pf_rw[(size_t) il + 1].data();
                        jb.pf_map  = map.data() + (size_t) (il + 1)*n_expert;
                        jb.pf_k    = lean_pf_k;
                        jb.pf_pool = lean_pf_pool;
                        jb.pf_hint = lean_pf_hint;
                        jb.pf_abort = moe_hybrid_pf_abort;
                        jb.pf_abort_ctx = &pf_ctx;
                        pf_job     = true;
                    }
                    lean_team->run(jb);
                    n_lean++;
                    if (memcmp(lean_y.data(), hg.out->data, (size_t) n_embd*sizeof(float)) != 0) {
                        if (n_lean_bad++ < 8) {
                            const float * ref = (const float *) hg.out->data;
                            int first = -1, cnt = 0;
                            for (int i = 0; i < n_embd; ++i) {
                                if (memcmp(&lean_y[(size_t) i], &ref[i], sizeof(float)) != 0) {
                                    cnt += 1;
                                    if (first < 0) {
                                        first = i;
                                    }
                                }
                            }
                            LLAMA_LOG_ERROR("moe_hybrid: LEAN SELF-CHECK MISMATCH at request %lld: layer %d nmiss %d, %d of %d floats differ, first at %d (lean %.9g, graph %.9g)\n",
                                            (long long) n_requests + 1, il, nmiss, cnt, n_embd, first, (double) lean_y[(size_t) first], (double) ref[first]);
                        }
                    }
                }
                memcpy(m->y, hg.out->data, (size_t) n_embd*sizeof(float));
            }
        }
        if (il >= 0 && il < n_layer) {
            m->prof[il][3] = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(moe_clock::now() - t0).count();
        }
        m->err = err;
        std::atomic_thread_fence(std::memory_order_release);
        m->res_seq = seq;
        served[j] = seq;
        sv_phase.store(0, std::memory_order_relaxed);

        const double us = moe_hybrid_us(t0);
        n_requests++;
        serve_sum += us;
        serve_max  = std::max(serve_max, us);
        if (us > 5000.0) {
            LLAMA_LOG_WARN("moe_hybrid: slow host request #%lld: layer %d token %d nmiss %d took %.1f ms (copy-in %.1f, graph compute %.1f, rest %.1f), dma_state %d\n",
                           (long long) n_requests, il, j, nmiss, us/1e3, in_us/1e3, comp_us/1e3, (us - in_us - comp_us)/1e3, dma_state.load());
        }
        nmiss_hist[err == 0 ? nmiss : 0]++;
        active = moe_clock::now();

        // G15: the reply is out; the workers spin until the GPU posts the next layer's request, so let them warm the caches for it (skipped when another request is already waiting)
        if (pf_job) {
            bool waiting = false;
            for (int k = 0; k < GGML_MOE_HYBRID_MAX_WIDTH; ++k) {
                waiting = waiting || mbs[k]->req_seq != served[k];
            }
            const auto tp0 = moe_clock::now();
            const bool ran = lean_team->post(!waiting);
            if (waiting) {
                pf_skipped++;
            } else if (ran) {
                const moe_lean::pf_choice & c = lean_team->last_choice();
                pf_last[(size_t) il + 1].n = c.n;
                for (int k = 0; k < c.n; ++k) {
                    pf_last[(size_t) il + 1].ids[k] = c.ids[k];
                }
                pf_last[(size_t) il + 1].tok = pf_tok_no;
                pf_trig++;
                pf_issued += c.n;
                pf_post_sum += moe_hybrid_us(tp0);
            }
            active = moe_clock::now();
        }

        // the token's inputs are uploaded by now: start the promotion copies queued by on_token
        int queued = 1;
        if (dma_thread_on) {
            if (dma_state.load(std::memory_order_relaxed) == 1 &&
                dma_state.compare_exchange_strong(queued, 3, std::memory_order_acquire)) {
                { std::lock_guard<std::mutex> lk(dma_mtx); } // order the notify after the waiter's predicate check
                dma_cv.notify_one();
            }
        } else if (dma_state.load(std::memory_order_relaxed) == 1 &&
            dma_state.compare_exchange_strong(queued, 2, std::memory_order_acquire)) {
            sv_t_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(moe_clock::now().time_since_epoch()).count(), std::memory_order_relaxed);
            sv_phase.store(2, std::memory_order_relaxed);
            sv_sub.store(21, std::memory_order_relaxed);
            submit_dma();
            sv_sub.store(22, std::memory_order_relaxed);
            ggml_backend_cuda_event_record_aux(cuda, ev);
            sv_sub.store(23, std::memory_order_relaxed);
            ggml_backend_cuda_aux_flush(cuda);
            sv_sub.store(24, std::memory_order_relaxed);
            n_dma_engine++;
            dma_submit_sum += moe_hybrid_us(active);
            dma_submit_max  = std::max(dma_submit_max, moe_hybrid_us(active));
            if (moe_hybrid_us(active) > 5000.0) {
                LLAMA_LOG_WARN("moe_hybrid: slow DMA submission #%lld took %.1f ms\n", (long long) n_dma_engine, moe_hybrid_us(active)/1e3);
            }
            dma_state.store(0, std::memory_order_release);
            sv_phase.store(0, std::memory_order_relaxed);
            active = moe_clock::now();
        }
    }
}

void moe_hybrid_engine::dma_loop() {
    std::unique_lock<std::mutex> lk(dma_mtx);
    for (;;) {
        dma_cv.wait(lk, [&] { return dma_stop || dma_state.load(std::memory_order_acquire) == 3; });
        if (dma_state.load(std::memory_order_acquire) == 3) {
            lk.unlock();
            const auto t0 = moe_clock::now();
            submit_dma();
            ggml_backend_cuda_event_record_aux(cuda, ev);
            ggml_backend_cuda_aux_flush(cuda);
            n_dma_engine++;
            const double us = moe_hybrid_us(t0);
            dma_submit_sum += us;
            dma_submit_max  = std::max(dma_submit_max, us);
            if (us > 5000.0) {
                LLAMA_LOG_WARN("moe_hybrid: slow DMA submission #%lld took %.1f ms (dma thread)\n", (long long) n_dma_engine, us/1e3);
            }
            dma_state.store(0, std::memory_order_release);
            lk.lock();
            continue;
        }
        if (dma_stop) {
            break;
        }
    }
}

// Opt-in diagnostics. Polls the mailbox every 2 ms; when the same request stays unanswered for wd_ms it reports once, with the
// phase of the host engine thread. A device-side wait times out (and traps the context) after 500 ms; this says WHERE the host was.
void moe_hybrid_engine::watchdog() {
    int32_t  last_req[GGML_MOE_HYBRID_MAX_WIDTH]  = {};
    moe_clock::time_point since[GGML_MOE_HYBRID_MAX_WIDTH];
    bool     reported[GGML_MOE_HYBRID_MAX_WIDTH]      = {};
    bool     reported_pend[GGML_MOE_HYBRID_MAX_WIDTH] = {};
    int64_t  nodes_at_since[GGML_MOE_HYBRID_MAX_WIDTH] = {};
    for (auto & s : since) {
        s = moe_clock::now();
    }
    uint64_t hb_last  = 0;
    auto     hb_t     = moe_clock::now();
    while (!wd_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        {   // keep the heartbeat age current even when no request is pending
            const uint64_t hb = sv_spins.load(std::memory_order_relaxed);
            if (hb != hb_last) {
                hb_last = hb;
                hb_t    = moe_clock::now();
            }
        }
        for (int j = 0; j < GGML_MOE_HYBRID_MAX_WIDTH; ++j) {
            ggml_moe_hybrid_mailbox * m = mbs[j];
            const int32_t req = m->req_seq;
            const int32_t res = m->res_seq;
            if (req != last_req[j]) {
                last_req[j] = req;
                since[j]    = moe_clock::now();
                reported[j] = false;
                reported_pend[j] = false;
                nodes_at_since[j] = ggml_backend_cuda_dbg_nodes_launched();
            }
            if (req == res) {
                continue;
            }
            const double pend_ms = moe_hybrid_us(since[j])/1e3;
            if (pend_ms > wd_max_ms) {
                wd_max_ms = pend_ms;
            }
            if (pend_ms >= 5.0 && !reported_pend[j]) {
                reported_pend[j] = true;
                wd_n_pend++;
            }
            if (pend_ms >= (double) wd_ms && !reported[j]) {
                reported[j] = true;
                const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(moe_clock::now().time_since_epoch()).count();
                // heartbeat age: how long ago the request loop last counted. Huge while the phase is 0 means the engine thread is NOT RUNNING
                // (descheduled); small means it is polling but does not see the request.
                const uint64_t hb_now = sv_spins.load(std::memory_order_relaxed);
                double hb_age_ms = 0.0;
                if (hb_now != hb_last) {
                    hb_last = hb_now;
                    hb_t    = moe_clock::now();
                } else {
                    hb_age_ms = moe_hybrid_us(hb_t)/1e3;
                }
                LLAMA_LOG_WARN("%s: moe_hybrid watchdog: token %d request %d unanswered for %.1f ms (res_seq %d, layer %d, nmiss %d); host engine phase %d.%d "
                               "for %.1f ms (last request seen %d), heartbeat age %.1f ms, dma_state %d, stop %d, wake_gen %u\n", __func__,
                               j, (int) req, pend_ms, (int) res, (int) m->layer, (int) m->nmiss, sv_phase.load(), sv_sub.load(),
                               (double) (now_ns - sv_t_ns.load())/1e6, (int) sv_seq.load(), hb_age_ms, dma_state.load(), (int) stop.load(),
                               (unsigned) wake_gen.load());
                LLAMA_LOG_WARN("%s: moe_hybrid watchdog (cont.): promotion copy index being submitted %d, graph nodes launched since the request went pending %lld, "
                               "main thread inside a graph compute %d\n", __func__, ggml_backend_cuda_dbg_aux_idx(),
                               (long long) (ggml_backend_cuda_dbg_nodes_launched() - nodes_at_since[j]), ggml_backend_cuda_dbg_in_compute());
            }
        }
    }
}

// The candidate scan of the policy: per layer the best uncached experts (EMA + tie-break, descending) against the worst cached ones (ascending), paired while the gain is
// positive, then the `budget` largest gains over all layers. Reads ema / tie / slot_of, writes only `out`.
// slots per layer for the current n_slot (see the members): the file's weights^alpha shared out over n_slot*n_layer slots, each layer within [split_min, split_max] and at most n_expert;
// the same procedure as split_from_weights() of tools/analyze-policy-replay.py, which scored it by exact replay of this policy
void moe_hybrid_engine::compute_split() {
    slots_l.assign((size_t) n_layer, n_slot);
    if (!split_path.empty()) {
        std::vector<double> w;
        if (FILE * f = fopen(split_path.c_str(), "r")) {
            double v = 0.0;
            while (w.size() < (size_t) n_layer && fscanf(f, "%lf", &v) == 1) {
                w.push_back(v);
            }
            fclose(f);
        }
        const int64_t total = (int64_t) n_slot*n_layer;
        const int     lo    = std::max(1, std::min(split_min, n_expert));
        const int     hi    = std::max(lo, std::min(split_max, n_expert));
        if ((int) w.size() != n_layer || total < (int64_t) lo*n_layer || total > (int64_t) hi*n_layer) {
            LLAMA_LOG_WARN("%s: moe_hybrid: LLAMA_MOE_HYBRID_SPLIT=%s unusable (%zu weights for %d layers, or %d slots per layer outside [%d, %d]); uniform split\n", __func__,
                           split_path.c_str(), w.size(), n_layer, n_slot, lo, hi);
        } else {
            std::vector<double> wa((size_t) n_layer);
            for (int l = 0; l < n_layer; ++l) {
                wa[(size_t) l] = std::pow(std::max(w[(size_t) l], 0.0) + 1e-9, split_alpha);
            }
            std::vector<double> alloc((size_t) n_layer, 0.0);
            std::vector<char>   active((size_t) n_layer, 1);
            double              free_slots = (double) (total - (int64_t) lo*n_layer);
            const double        cap = (double) (hi - lo);
            for (int iter = 0; iter < 20; ++iter) {
                double sum = 0.0;
                for (int l = 0; l < n_layer; ++l) {
                    sum += active[(size_t) l] ? wa[(size_t) l] : 0.0;
                }
                std::vector<double> want((size_t) n_layer, 0.0);
                bool any_over = false;
                for (int l = 0; l < n_layer; ++l) {
                    if (active[(size_t) l] && sum > 0.0) {
                        want[(size_t) l] = wa[(size_t) l]/sum*free_slots;
                        any_over = any_over || want[(size_t) l] > cap - alloc[(size_t) l];
                    }
                }
                if (!any_over) {
                    for (int l = 0; l < n_layer; ++l) {
                        alloc[(size_t) l] += want[(size_t) l];
                    }
                    break;
                }
                double used = 0.0;
                for (int l = 0; l < n_layer; ++l) {
                    if (active[(size_t) l] && want[(size_t) l] > cap - alloc[(size_t) l]) {
                        alloc[(size_t) l] = cap;
                        active[(size_t) l] = 0;
                    }
                    used += alloc[(size_t) l];
                }
                free_slots = (double) (total - (int64_t) lo*n_layer) - used;
            }
            int64_t sum_i = 0;
            std::vector<double> frac((size_t) n_layer);
            for (int l = 0; l < n_layer; ++l) {
                const double s = (double) lo + alloc[(size_t) l];
                slots_l[(size_t) l] = (int) std::floor(s);
                frac[(size_t) l] = s - std::floor(s);
                sum_i += slots_l[(size_t) l];
            }
            std::vector<int> ord((size_t) n_layer);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return frac[(size_t) a] > frac[(size_t) b]; });
            for (int64_t i = 0, rem = total - sum_i; i < rem; ++i) {
                slots_l[(size_t) ord[(size_t) (i % n_layer)]] += 1;
            }
            int mn = 1 << 30, mx = 0;
            for (int l = 0; l < n_layer; ++l) {
                slots_l[(size_t) l] = std::min(slots_l[(size_t) l], n_expert);
                mn = std::min(mn, slots_l[(size_t) l]);
                mx = std::max(mx, slots_l[(size_t) l]);
            }
            LLAMA_LOG_INFO("%s: moe_hybrid: non-uniform slot split from %s (alpha %.2f): %d .. %d slots per layer, %d on average\n", __func__, split_path.c_str(), split_alpha, mn, mx, n_slot);
        }
    }
    slot_off.assign((size_t) n_layer + 1, 0);
    for (int l = 0; l < n_layer; ++l) {
        slot_off[(size_t) l + 1] = slot_off[(size_t) l] + (size_t) slots_l[(size_t) l];
    }
}

void moe_hybrid_engine::compute_pairs(std::vector<moe_hybrid_pair> & out) const {
    out.clear();
    for (int l = 0; l < n_layer; ++l) {
        double cs[MOE_HYBRID_K]; int ce[MOE_HYBRID_K]; int nc = 0; // best uncached, score desc
        double vs[MOE_HYBRID_K]; int ve[MOE_HYBRID_K]; int nv = 0; // worst cached, score asc
        const size_t base = (size_t) l*n_expert;
        for (int e = 0; e < n_expert; ++e) {
            const double s = ema[base + e] + tie[base + e];
            if (slot_of[base + e] < 0) {
                if (recency && !last_sel[base + e]) {
                    continue;
                }
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
            out.push_back({ gain, l, ce[k], ve[k] });
        }
    }
    const int cap = budget > 0 ? budget*std::max(1, scan_mult) : budget;
    if (cap > 0 && (int) out.size() > cap) {
        std::stable_sort(out.begin(), out.end(), [](const moe_hybrid_pair & a, const moe_hybrid_pair & b) {
            return a.gain > b.gain;
        });
        out.resize(cap);
    }
}

void moe_hybrid_engine::scan_loop() {
#ifdef _WIN32
    if (scan_cpu >= 0 && scan_cpu < 64) {
        if (SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << scan_cpu) == 0) {
            LLAMA_LOG_WARN("%s: moe_hybrid: scan helper could not be pinned to CPU %d (%lu), left unpinned\n", __func__, scan_cpu, (unsigned long) GetLastError());
        }
    }
#endif
    std::unique_lock<std::mutex> lk(scan_mtx);
    for (;;) {
        scan_cv.wait(lk, [&] { return scan_stop || scan_req; });
        if (scan_req) {
            scan_req = false;
            lk.unlock();
            compute_pairs(scan_out);
            scan_busy.store(0, std::memory_order_release);
            lk.lock();
            continue;
        }
        if (scan_stop) {
            break;
        }
    }
}

// asks for a scan against the current state; in mode 1 the same scan runs inline (the reference), in mode 2 on the helper
void moe_hybrid_engine::start_scan() {
    scan_have = true;
    if (scan_mode == 1 || !scan_th.joinable()) {
        compute_pairs(scan_out);
        return;
    }
    scan_busy.store(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(scan_mtx);
        scan_req = true;
    }
    scan_cv.notify_one();
}

// G31s: fold the routing of K prompt tokens into the EMA as K decoded tokens would (ids: [K][n_layer][n_used], -1 = no entry)
void moe_hybrid_engine::seed(const int32_t * ids, int K) {
    if (K <= 0) {
        return;
    }
    join_scan();   // the helper's scan of the previous token reads the EMA
    const double decay = 1.0 - lambda;
    double scale = 1.0;
    for (int r = 0; r < K; ++r) {
        scale *= lambda;
    }
    for (double & v : ema) {
        v *= scale;
    }
    if (recency) {
        std::fill(last_sel.begin(), last_sel.end(), (uint8_t) 0);
    }
    for (int r = 0; r < K; ++r) {
        double wgt = decay;
        for (int q = r + 1; q < K; ++q) {
            wgt *= lambda;
        }
        for (int l = 0; l < n_layer; ++l) {
            for (int k = 0; k < n_used; ++k) {
                const int e = ids[((size_t) r*n_layer + l)*n_used + k];
                if (e < 0 || e >= n_expert) {
                    continue;
                }
                const size_t i = (size_t) l*n_expert + e;
                ema[i] += wgt;
                if (recency && r == K - 1) {
                    last_sel[i] = 1;
                }
            }
        }
    }
}

void moe_hybrid_engine::join_scan() {
    if (!scan_th.joinable()) {
        return;
    }
    uint32_t spins = 0;
    while (scan_busy.load(std::memory_order_acquire) != 0) {
        MOE_HYBRID_PAUSE();
        if ((++spins & 0xFFFF) == 0) {
            std::this_thread::yield();
        }
    }
}

void moe_hybrid_engine::on_token(int n_new) {
#ifdef _WIN32
    if (main_cpu >= 0 && !main_pinned) {
        main_pinned = true;
        if (main_cpu < 64 && SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << main_cpu) != 0) {
            LLAMA_LOG_INFO("%s: moe_hybrid: the decoding thread is pinned to logical CPU %d\n", __func__, main_cpu);
        } else {
            LLAMA_LOG_WARN("%s: moe_hybrid: could not pin the decoding thread to logical CPU %d\n", __func__, main_cpu);
        }
    }
#endif
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

    // the helper's scan of the previous token must be finished before the EMA (which it reads) changes
    {
        const auto tj = moe_clock::now();
        join_scan();
        ot_join += moe_hybrid_us(tj);
    }

    const auto tA = moe_clock::now();
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
            for (int j = 0; j < GGML_MOE_HYBRID_MAX_WIDTH; ++j) {
                memset(mbs[j]->prof, 0, sizeof(mbs[j]->prof));
            }
        }
    }
    const auto tB = moe_clock::now();
    int  hits    = 0;
    int  lookups = 0;
    bool have    = false;
    const double decay = 1.0 - lambda;
    // rows: the token rows of the graph that just ran (one for a plain decode step, 1 + drafts for an MTP verify step). A row counts when any layer wrote it. The EMA takes
    // them in order, ema = lambda^rows*ema + sum_r decay*lambda^(rows-1-r)*[selected in row r], which for one row is exactly the single-token update.
    int rows_valid = 0;
    {
        bool row_has[GGML_MOE_HYBRID_MAX_WIDTH] = {};
        for (int r = 0; r < prev_w && r < GGML_MOE_HYBRID_MAX_WIDTH; ++r) {
            const int32_t * tb = trace + (size_t) r*trace_stride;
            for (int l = 0; l < n_layer; ++l) {
                if (tb[(size_t) l*GGML_MOE_HYBRID_MAX_USED] >= 0) {
                    row_has[r] = true;
                    break;
                }
            }
            rows_valid += row_has[r] ? 1 : 0;
        }
        if (rows_valid > 0) {
            have = true;
            double scale = 1.0;
            for (int r = 0; r < rows_valid; ++r) {
                scale *= lambda;
            }
            for (double & v : ema) {
                v *= scale;
            }
            if (recency) {
                std::fill(last_sel.begin(), last_sel.end(), (uint8_t) 0);
            }
            int ri = 0;
            for (int r = 0; r < prev_w && r < GGML_MOE_HYBRID_MAX_WIDTH; ++r) {
                if (!row_has[r]) {
                    continue;
                }
                double wgt = decay;
                for (int q = ri + 1; q < rows_valid; ++q) {
                    wgt *= lambda;
                }
                const int32_t * tb = trace + (size_t) r*trace_stride;
                if (trace_fp != nullptr) {
                    const int32_t tok = (int32_t) (n_rows + ri);
                    fwrite(&tok, sizeof(int32_t), 1, trace_fp);
                    fwrite(tb, sizeof(int32_t), (size_t) n_layer*GGML_MOE_HYBRID_MAX_USED, trace_fp);
                }
                ++ri;
                for (int l = 0; l < n_layer; ++l) {
                    const int32_t * tr = tb + (size_t) l*GGML_MOE_HYBRID_MAX_USED;
                    if (tr[0] < 0) {
                        continue;
                    }
                    for (int k = 0; k < n_used; ++k) {
                        const int e = tr[k];
                        if (e < 0 || e >= n_expert) {
                            GGML_ABORT("moe_hybrid: corrupt routing trace (token row %d, layer %d, entry %d = %d)", r, l, k, e);
                        }
                        const size_t i = (size_t) l*n_expert + e;
                        lookups++;
                        hits += map[i] >= 0;
                        ema[i] += wgt;
                        if (recency) {
                            last_sel[i] = 1;
                        }
                    }
                }
            }
        }
    }
    std::fill(trace, trace + (size_t) GGML_MOE_HYBRID_MAX_WIDTH*trace_stride, -1);
    prev_w = std::max(1, std::min(n_new, (int) GGML_MOE_HYBRID_MAX_WIDTH));
    scan_mult = prev_w;

    const auto tC = moe_clock::now();
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

    const auto tD = moe_clock::now();
    auto tE = tD;
    auto tF = tD;
    if (have && !waited && budget != 0) {
        if (scan_mode == 0) {
            compute_pairs(pairs);
        } else {
            // lag 1: the pairs scanned at the end of the previous on_token. Nothing but this loop changes slot_of, and the pending promotions were published above,
            // so the victims are still cached (and published) and the candidates still uncached.
            pairs.clear();
            if (scan_have) {
                pairs.swap(scan_out);
            }
            scan_have = false;
        }
        tE = moe_clock::now();
        for (const moe_hybrid_pair & c : pairs) {
            const size_t iv = (size_t) c.l*n_expert + c.victim;
            const size_t ic = (size_t) c.l*n_expert + c.cand;
            const int    s  = slot_of[iv];
            GGML_ASSERT(s >= 0 && map[iv] == s && slot_of[ic] < 0);
            map[iv]     = -1;
            slot_of[iv] = -1;
            slot_of[ic] = s;
            slot_expert[slot_off[(size_t) c.l] + s] = c.cand;
            pending.push_back({ c.l, c.cand, s });
            queue_expert(c.l, c.cand, s);
        }
        if (!pairs.empty()) {
            dma_state.store(1, std::memory_order_release);
        }
        n_prom = (int) pairs.size();
        tF = moe_clock::now();
    }
    if (scan_mode != 0 && have && budget != 0) {
        start_scan();   // against the state after this apply, for the next token
        tF = moe_clock::now();
    }
    if (xd_ntok > 0) {
        const int32_t idx = (int32_t) (n_rows + (have ? rows_valid : 0));   // the stats block below adds rows_valid; the step about to run is the next row
        ggml_backend_tensor_set(state, &idx, 12*sizeof(int32_t), sizeof(int32_t));
    }

    const double us = moe_hybrid_us(t0);
    if (have) {
        auto dur = [](moe_clock::time_point a, moe_clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        ot_sync     += dur(t0, tA);
        ot_timeline += dur(tA, tB);
        ot_ema      += dur(tB, tC);
        ot_pending  += dur(tC, tD);
        ot_scan     += dur(tD, tE);
        ot_apply    += dur(tE, tF);
    }
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
        n_rows       += rows_valid;
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
    wd_stop.store(true);
    if (scan_th.joinable()) {
        {
            std::lock_guard<std::mutex> lk(scan_mtx);
            scan_stop = true;
        }
        scan_cv.notify_one();
        scan_th.join();
    }
    if (wd_th.joinable()) {
        wd_th.join();
        LLAMA_LOG_INFO("%s: moe_hybrid watchdog summary: %lld requests observed pending for 5 ms or more, worst observed pending %.1f ms\n",
                       __func__, (long long) wd_n_pend, wd_max_ms);
    }
    if (th.joinable()) {
        stop.store(true);
        wake();
        th.join();
    }
    if (dma_th.joinable()) {
        {
            std::lock_guard<std::mutex> lk(dma_mtx);
            dma_stop = true;
        }
        dma_cv.notify_one();
        dma_th.join();
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
        if (n_tokens > 0) {
            LLAMA_LOG_INFO("moe_hybrid: on_token breakdown (scan mode %d), avg us per token: sync + dma wait %.1f (of which helper join %.1f), timeline %.1f, trace + ema %.1f, pending check %.1f, candidate scan %.1f, apply + queue %.1f\n",
                           scan_mode, ot_sync/n_tokens, ot_join/n_tokens, ot_timeline/n_tokens, ot_ema/n_tokens, ot_pending/n_tokens, ot_scan/n_tokens, ot_apply/n_tokens);
        }
        LLAMA_LOG_INFO("moe_hybrid: tokens %lld, hit rate %.4f (%lld/%lld), promotions/token %.2f, pending waits %lld, "
                       "on_token avg %.1f max %.1f us | host requests %lld, serve avg %.1f max %.1f us, nmiss hist [err,1..%d] %s | "
                       "promotion batches engine %lld (submit avg %.1f max %.1f us) main %lld\n",
                       (long long) n_tokens, n_lookups ? (double) n_hits/n_lookups : 0.0, (long long) n_hits, (long long) n_lookups,
                       n_tokens ? (double) n_promotions/n_tokens : 0.0, (long long) n_waits,
                       n_tokens ? on_token_sum/n_tokens : 0.0, on_token_max,
                       (long long) n_requests, n_requests ? serve_sum/n_requests : 0.0, serve_max, n_used, hist,
                       (long long) n_dma_engine, n_dma_engine ? dma_submit_sum/n_dma_engine : 0.0, dma_submit_max, (long long) n_dma_main);
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
        if (lean_mode > 0) {
            LLAMA_LOG_INFO("moe_hybrid: lean host leg mode %d: %lld requests through the lean team%s", lean_mode, (long long) n_lean,
                           lean_mode == 2 ? "" : "\n");
            if (lean_mode == 2) {
                LLAMA_LOG_INFO(", each compared bit for bit with the graph: %lld DIFFER\n", (long long) n_lean_bad);
            }
            if (lean_pf_k > 0) {
                LLAMA_LOG_INFO("moe_hybrid: lean prefetch K=%d: %lld post stages (%lld skipped because another request was waiting), %.1f us each on thread 0; %lld experts chosen, %lld of them "
                               "were misses of the next request (precision %.3f); %lld requests had a prediction, %lld of them contained a prefetched expert, %.3f of their %lld misses were prefetched; "
                               "useful prefetches per token %.2f\n",
                               lean_pf_k, (long long) pf_trig, (long long) pf_skipped, pf_trig ? pf_post_sum/pf_trig : 0.0, (long long) pf_issued, (long long) pf_useful,
                               pf_issued ? (double) pf_useful/pf_issued : 0.0, (long long) pf_req_with_set, (long long) pf_req_hit,
                               pf_miss_in_set_reqs ? (double) pf_useful/pf_miss_in_set_reqs : 0.0, (long long) pf_miss_in_set_reqs, n_tokens ? (double) pf_useful/n_tokens : 0.0);
            }
            if (lean_prof && lean_stats.n > 0) {
                LLAMA_LOG_INFO("moe_hybrid: lean team stages (thread 0's view, mean per request over %lld requests, us): quantize x %.2f, up+gate %.2f, swiglu+quantize %.2f, down+sum %.2f\n",
                               (long long) lean_stats.n, lean_stats.a_us/lean_stats.n, lean_stats.b_us/lean_stats.n, lean_stats.c_us/lean_stats.n, lean_stats.d_us/lean_stats.n);
            }
        }
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
    if (trace_fp) {
        fclose(trace_fp);
        trace_fp = nullptr;
    }
    if (ev) {
        ggml_backend_event_free(ev);
        ev = nullptr;
    }
    if (xd_buf) {
        const int rec = (int) std::min<int64_t>(n_rows, xd_ntok);
        if (FILE * f = fopen(xd_path.c_str(), "wb")) {
            const int32_t hdr[4] = { rec, n_layer, n_embd, 0 };
            fwrite("MOEXD001", 1, 8, f);
            fwrite(hdr, sizeof(int32_t), 4, f);
            fwrite(xd_host, sizeof(float), (size_t) rec*n_layer*n_embd, f);
            fclose(f);
            LLAMA_LOG_INFO("%s: moe_hybrid: wrote the FFN inputs of %d decode steps to %s\n", __func__, rec, xd_path.c_str());
        } else {
            LLAMA_LOG_WARN("%s: moe_hybrid: cannot write %s\n", __func__, xd_path.c_str());
        }
        ggml_backend_cuda_moe_hybrid_set_xdump(nullptr, 0);
        ggml_backend_buffer_free(xd_buf);
        xd_buf = nullptr;
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

// published slot map, refreshed (and the policy stepped) once per decoded step (one token, or a verify step of 1 + drafts tokens)
class llm_graph_input_moe_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_moe_hybrid(moe_hybrid_engine * engine, int n_tokens) : eng(engine), n_tok(n_tokens) {}

    void set_input(const llama_ubatch * ubatch) override {
        eng->on_token((int) ubatch->n_tokens);
        ggml_backend_tensor_set(map, eng->map.data(), 0, ggml_nbytes(map));
    }

    // a graph is built for one width; any other width is a different graph (its own publish / wait ops and mailbox rows)
    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.n_tokens == (uint32_t) n_tok;
    }

    moe_hybrid_engine * eng;
    int                 n_tok;
    ggml_tensor *       map = nullptr; // I32 [n_layer*n_expert]
};

} // namespace

bool llama_moe_hybrid_seed_impl(const int32_t * ids, int K, int * n_layer, int * n_used) {
    moe_hybrid_engine * eng = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        if (g_moe_hybrid_engines.size() != 1) {
            return false;
        }
        eng = g_moe_hybrid_engines.begin()->second.get();
    }
    if (ids == nullptr) {
        // the caller asks for the geometry
        if (n_layer != nullptr) {
            *n_layer = eng->n_layer;
        }
        if (n_used != nullptr) {
            *n_used = eng->n_used;
        }
        return true;
    }
    eng->seed(ids, K);
    return true;
}

void llama_moe_hybrid_set_keep_stats(bool keep) {
    g_moe_hybrid_keep_stats.store(keep);
}

bool llama_moe_hybrid_cache_info_impl(int64_t * cache_bytes, int * n_slot) {
    moe_hybrid_engine * eng = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        if (g_moe_hybrid_engines.size() != 1) {
            return false;
        }
        eng = g_moe_hybrid_engines.begin()->second.get();
    }
    const int slots = eng->info_n_slot.load();
    if (slots <= 0) {
        return false;   // no cache yet, or resize() is rebuilding it
    }
    *cache_bytes = eng->info_cache_bytes.load();
    *n_slot      = slots;
    return true;
}

bool llama_moe_hybrid_make_room_impl(int64_t need_free_bytes, int64_t slack_bytes, int min_slots, bool * shrunk) {
    if (shrunk != nullptr) {
        *shrunk = false;
    }
    moe_hybrid_engine * eng = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        if (g_moe_hybrid_engines.size() != 1) {
            return false;
        }
        eng = g_moe_hybrid_engines.begin()->second.get();
    }
    if (eng->cache_buf == nullptr || eng->cuda == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(eng->cuda);
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    if ((int64_t) free_b >= need_free_bytes) {
        return true;
    }
    const int64_t cache_bytes = (int64_t) ggml_backend_buffer_get_size(eng->cache_buf);
    const int64_t row_bytes   = cache_bytes / (eng->n_slot + 1);
    if (row_bytes <= 0) {
        return false;
    }
    const int64_t deficit   = need_free_bytes - (int64_t) free_b + slack_bytes;
    const int64_t drop_rows = (deficit + row_bytes - 1) / row_bytes;
    const int     new_slots = (int) std::max<int64_t>(std::max(1, min_slots), (int64_t) eng->n_slot - drop_rows);
    if (new_slots >= eng->n_slot) {
        return true;   // already at the floor: the caller tries the commit anyway
    }
    ggml_backend_synchronize(eng->cuda);   // a graph of the ubatch before this one may still read the cache
    // WARN so that it shows in the server log (the library's INFO lines are filtered there); a shrink is rare: it follows the growth of the context
    LLAMA_LOG_WARN("%s: moe_hybrid: device memory is short by %.1f MiB (a KV commit): expert cache %d -> %d slots per layer\n", __func__, (double) (need_free_bytes - (int64_t) free_b)/1048576.0, eng->n_slot, new_slots);
    if (!llama_moe_hybrid_resize_impl(new_slots)) {
        return false;
    }
    if (shrunk != nullptr) {
        *shrunk = true;
    }
    return true;
}

// G31: size the cache to a keep-free target (the rule of the runner's tg_transition(), plus shrinking). All sizes in bytes; slots clamp to [min_slots, min(max_slots, n_expert)].
bool llama_moe_hybrid_auto_resize_impl(int64_t keep_free_bytes, int64_t safety_bytes, int min_slots, int max_slots, int * slots_before, int * slots_after, int64_t * free_before_bytes) {
    moe_hybrid_engine * eng = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        if (g_moe_hybrid_engines.size() != 1) {
            return false;
        }
        eng = g_moe_hybrid_engines.begin()->second.get();
    }
    if (eng->cache_buf == nullptr || eng->cuda == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(eng->cuda);
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    const int64_t cache_bytes = (int64_t) ggml_backend_buffer_get_size(eng->cache_buf);
    const int64_t row_bytes   = cache_bytes / (eng->n_slot + 1);
    if (row_bytes <= 0) {
        return false;
    }
    const int64_t room = (int64_t) free_b + cache_bytes - keep_free_bytes - safety_bytes;
    int64_t slots = room / row_bytes - 1;
    slots = std::min<int64_t>(slots, std::min<int64_t>(max_slots, eng->n_expert));
    slots = std::max<int64_t>(slots, std::max(1, min_slots));
    if (slots_before != nullptr) {
        *slots_before = eng->n_slot;
    }
    if (free_before_bytes != nullptr) {
        *free_before_bytes = (int64_t) free_b;
    }
    if (slots_after != nullptr) {
        *slots_after = (int) slots;
    }
    if ((int) slots == eng->n_slot) {
        return true;
    }
    return llama_moe_hybrid_resize_impl((int) slots);
}

bool llama_moe_hybrid_resize_impl(int n_slot) {
    moe_hybrid_engine * eng = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_hybrid_mtx);
        if (g_moe_hybrid_engines.size() != 1) {
            return false;
        }
        eng = g_moe_hybrid_engines.begin()->second.get();
    }
    if (!eng->resize(n_slot)) {
        return false;
    }
    char v[16];
    snprintf(v, sizeof(v), "%d", n_slot);
#ifdef _WIN32
    _putenv_s("LLAMA_MOE_HYBRID", v);
#else
    setenv("LLAMA_MOE_HYBRID", v, 1);
#endif
    return true;
}

ggml_tensor * llama_moe_hybrid_build_ffn(const llm_graph_context & g, const llama_model & model, int il,
                                         ggml_tensor * cur, ggml_tensor ** pub) {
    *pub = nullptr;

    const int     n_slot = moe_hybrid_env_int("LLAMA_MOE_HYBRID", 0);
    const int64_t n_tok  = g.n_tokens;   // 1 for a decode step; 1 + drafts for an MTP verify step
    if (n_slot <= 0 || n_tok < 1 || n_tok > GGML_MOE_HYBRID_MAX_WIDTH || cur->ne[1] != n_tok || g.layer_streaming != nullptr || il < 0 || il >= g.n_layer ||
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
        auto owned = std::make_unique<llm_graph_input_moe_hybrid>(eng, (int) n_tok);
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
    ggml_tensor * logits = g.build_lora_mm(L.ffn_gate_inp, cur); // [n_expert, T]
    g.cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    g.cb(probs, "ffn_moe_probs", il);

    ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, n_expert_used); // [n_expert_used, T]
    g.cb(selected->src[0], "ffn_moe_argsort", il);
    g.cb(selected, "ffn_moe_topk", il);

    probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tok);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected); // [1, n_expert_used, T]
    g.cb(weights, "ffn_moe_weights", il);

    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tok);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
    g.cb(weights_sum, "ffn_moe_weights_sum", il);
    weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
    g.cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
    weights = ggml_div(ctx0, weights, weights_sum);
    g.cb(weights, "ffn_moe_weights_norm", il);
    weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tok);

    const float w_scale = g.hparams.expert_weights_scale;
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        g.cb(weights, "ffn_moe_weights_scaled", il);
    }
    ggml_build_forward_expand(g.gf, weights);

    // cache slots for the selected experts (miss -> zero slot) and the host requests for the misses, one mailbox per token
    ggml_tensor * p = ggml_moe_hybrid_publish(ctx0, selected, weights, cur, inp->map, eng->state, il,
                                              (int32_t) n_expert, eng->slots_l[(size_t) il], eng->mb_dev, (int32_t) eng->mb_stride, eng->trace_dev, eng->trace_stride);
    g.cb(p, "ffn_moe_hybrid_pub", il);
    ggml_build_forward_expand(g.gf, p);

    ggml_tensor * ids  = ggml_view_2d(ctx0, p, n_expert_used, n_tok, p->nb[1], 0);
    ggml_tensor * cur3 = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tok);

    // every miss in `ids` is the zero slot, so a token's ids repeat a slot; the CUDA dispatch must keep these nodes on the kernels that read one id per output
    // (floating-point banks, e.g. the bf16 layers of the Q8_K_XL quants, would take mmf, which writes one output per (token, expert) and leaves the rest unwritten)
    auto mark_dup_ids = [](ggml_tensor * mm) {
        ((int32_t *) mm->op_params)[GGML_MOE_HYBRID_ID_FLAG_IDX] = GGML_MOE_HYBRID_ID_FLAG_DUP;
        return mm;
    };
    ggml_tensor * up = mark_dup_ids(ggml_mul_mat_id(ctx0, eng->c_up[il], cur3, ids)); // [n_ff, n_expert_used, T]
    g.cb(up, "ffn_moe_up", il);
    ggml_tensor * gate = mark_dup_ids(ggml_mul_mat_id(ctx0, eng->c_gate[il], cur3, ids));
    g.cb(gate, "ffn_moe_gate", il);
    ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
    g.cb(act, "ffn_moe_swiglu", il);
    ggml_tensor * experts = mark_dup_ids(ggml_mul_mat_id(ctx0, eng->c_down[il], act, ids)); // [n_embd, n_expert_used, T]
    g.cb(experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);
    g.cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(g.gf, experts);

    ggml_tensor * views[GGML_MOE_HYBRID_MAX_USED];
    for (uint32_t i = 0; i < g.hparams.n_expert_used; ++i) {
        views[i] = ggml_view_2d(ctx0, experts, n_embd, n_tok, experts->nb[2], i*experts->nb[1]);
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
