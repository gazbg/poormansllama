#pragma once

// G14: a lean host leg for the hybrid MoE decode. The host serves the experts that missed the VRAM cache. The graph that did this (three mul_mat_id, a swiglu, a multiply and a
// chain of adds, 8-12 nodes) measured 33 us of fixed cost per request on top of 43 us per expert (the DRAM-bound part), because every node ends in a thread-pool barrier and
// every mul_mat_id rebuilds its row mapping over all 256 experts. This file computes the same result with the same vec_dot kernels, quantizers and swiglu routine the graph calls
// (so the output is bit-identical, which tests/test-moe-lean-expert.cpp checks), in ONE parallel region with three spin barriers:
//   A  quantize the input row to the dot-product type (each thread its blocks)          barrier
//   B  up and gate dot products, each thread its slice of rows, for every missed expert barrier
//   C  every thread: swiglu of the full [m][n_ff] and the quantization of the result (redundant on purpose: ~1 us, cheaper than a barrier)
//   D  down dot products for the thread's slice of output rows, the router weights and the sum over the experts, written straight into y     barrier
// The summation order is the graph's: out = ((e0*w0 + e1*w1) + e2*w2) + ..., each product and each sum rounded to float (the products are stored before the sums so that no compiler
// can contract a multiply and the following add into one fused operation: the graph rounds between them, because they are separate ops).
//
// G15: an optional stage after the reply (`team::post`). The engine has just answered the request of layer l-1 and the GPU still needs ~90 us before it posts layer l's: the workers
// spin through it. They use it to predict layer l's misses from the request's input row (the router of layer l on the FFN input of layer l-1, rows in q8_0, one barrier), take the
// first K experts of the predicted top ranks that are not resident and PREFETCH their own row slices of those experts (the same slices B and D read) into their caches, so that the
// request of layer l reads the experts that were predicted right from the cache instead of from DRAM. A prefetch only touches memory: it cannot change a result.
//
// Several token rows (an MTP verify step posts one request per row for the same layer) can be served as ONE job: each expert that any row missed is read once, and each of its
// weight rows is used for every token row that selected it while it is still in L1. Each row keeps its own expert order for the sum, so its result is the one a job of that row
// alone gives, bit for bit.
//
// Header only; depends on ggml.h / ggml-cpu.h. `team` owns the worker threads (the caller of run() is thread 0).

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#define MOE_LEAN_PAUSE() _mm_pause()
#else
#define MOE_LEAN_PAUSE() std::this_thread::yield()
#endif

namespace moe_lean {

// time stamp counter and the logical CPU the thread runs on (IA32_TSC_AUX holds the processor number on Windows and Linux), for the stall diagnostic
inline uint64_t tsc_now(unsigned * cpu) {
#if defined(_M_X64) || defined(__x86_64__)
    unsigned       aux = 0;
    const uint64_t t   = __rdtscp(&aux);
    if (cpu) { *cpu = aux & 0xFFFu; }
    return t;
#else
    if (cpu) { *cpu = 0; }
    return (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

constexpr int MAX_M  = 8; // experts per request (n_expert_used)
constexpr int MAX_PF = 4; // experts prefetched per request
constexpr int MAX_R  = 4; // token rows per job
constexpr int MAX_U  = MAX_R*MAX_M;

// one routed-expert bank of a layer: ne = [k, rows, n_expert], plain contiguous host memory
struct bank {
    const char * data         = nullptr;
    size_t       row_bytes    = 0; // nb[1]
    size_t       expert_bytes = 0; // nb[2]
    ggml_type    type         = GGML_TYPE_F32;
    int64_t      k = 0, rows = 0;
};

inline bank bank_of(const ggml_tensor * t) {
    bank b;
    b.data         = (const char *) t->data;
    b.row_bytes    = t->nb[1];
    b.expert_bytes = t->nb[2];
    b.type         = t->type;
    b.k            = t->ne[0];
    b.rows         = t->ne[1];
    return b;
}

struct layer_banks {
    bank up, gate, down;
};

struct row {
    int             m   = 0;
    const int32_t * ids = nullptr; // [m]
    const float *   w   = nullptr; // [m]
    const float *   x   = nullptr; // [n_embd]
    float *         y   = nullptr; // [n_embd]
};

struct job {
    const layer_banks * lb  = nullptr;
    int                 m   = 0;
    const int32_t *     ids = nullptr; // [m]
    const float *       w   = nullptr; // [m]
    const float *       x   = nullptr; // [n_embd]
    float *             y   = nullptr; // [n_embd]

    // nr > 0: serve rows[0 .. nr) together (m .. y above are then not used)
    int nr = 0;
    row rows[MAX_R];

    // set by run(): the distinct experts of the job; u_pair holds r*MAX_M + e for each row r that has expert u_id at its position e
    int     n_u = 0;
    int32_t u_id[MAX_U]          = {};
    uint8_t u_n[MAX_U]           = {};
    uint8_t u_pair[MAX_U][MAX_R] = {};

    // G15 (optional): after the reply, predict the NEXT layer's misses from this row and prefetch them. pf_lb == nullptr or pf_k == 0 = no such stage.
    const layer_banks * pf_lb    = nullptr; // banks of the next layer
    const char *        pf_rw    = nullptr; // router of the next layer, q8_0 rows [n_expert][row_size(q8_0, n_embd)]
    const int32_t *     pf_map   = nullptr; // slot per expert of the next layer, < 0 = not resident (read racily: a stale value only changes which experts are prefetched)
    int                 pf_k     = 0;       // experts to prefetch (<= MAX_PF)
    int                 pf_pool  = 12;      // candidates come from the predictor's top-pool ranks
    int                 pf_hint  = 1;       // 0 prefetcht0, 1 prefetcht1 (L2), 2 prefetcht2, 3 prefetchnta, 4 plain loads of every line (a prefetch instruction can be dropped when the core's
                                            // miss buffers are full; a load waits for them, so the stream of one thread runs at that thread's bandwidth)
    const int32_t *     pf_force = nullptr; // tests: prefetch exactly these pf_k experts, no prediction
    bool              (*pf_abort)(void *) = nullptr; // called every 128 lines by every thread: true = stop the prefetch (the engine says: another request is waiting)
    void *              pf_abort_ctx = nullptr;
};

// the shapes the lean path computes; anything else must go through the graph
inline bool supported(const layer_banks & lb) {
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(lb.up.type);
    const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(lb.gate.type);
    const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(lb.down.type);
    if (!tu || !tg || !td || !tu->vec_dot || !tg->vec_dot || !td->vec_dot) {
        return false;
    }
    if (tu->vec_dot_type != tg->vec_dot_type || tu->nrows != 1 || tg->nrows != 1 || td->nrows != 1) {
        return false;
    }
    const ggml_type_traits_cpu * qx = ggml_get_type_traits_cpu(tu->vec_dot_type);
    const ggml_type_traits_cpu * qa = ggml_get_type_traits_cpu(td->vec_dot_type);
    if (!qx || !qa || !qx->from_float || !qa->from_float) {
        return false;
    }
    if (lb.up.type != lb.gate.type || lb.up.row_bytes != lb.gate.row_bytes || lb.up.expert_bytes != lb.gate.expert_bytes) {
        return false;
    }
    return lb.up.k == lb.gate.k && lb.up.rows == lb.gate.rows && lb.down.k == lb.up.rows && lb.up.k % ggml_blck_size(tu->vec_dot_type) == 0 &&
           lb.down.k % ggml_blck_size(td->vec_dot_type) == 0 && lb.down.rows == lb.up.k;
}

// q8_0 copy of a router matrix given as float rows [n_expert][n_embd] (n_embd must be a multiple of 32): the predictor's weights, 8.5 bits per weight
inline std::vector<char> quantize_router_q8(const float * rows, int64_t n_expert, int64_t n_embd) {
    const ggml_type_traits_cpu * t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const size_t                 rr = ggml_row_size(GGML_TYPE_Q8_0, n_embd);
    std::vector<char>            out((size_t) n_expert*rr);
    for (int64_t e = 0; e < n_expert; ++e) {
        t8->from_float(rows + (size_t) e*n_embd, out.data() + (size_t) e*rr, n_embd);
    }
    return out;
}

// sense-reversing spin barrier
struct barrier {
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
    int n = 1;

    void wait(int & local_sense) {
        local_sense ^= 1;
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(local_sense, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) != local_sense) {
                MOE_LEAN_PAUSE();
            }
        }
    }
};

// a 64-byte aligned scratch buffer
struct abuf {
    std::vector<char> v;
    char *            p = nullptr;

    void alloc(size_t n) {
        v.assign(n + 64, 0);
        p = (char *) (((uintptr_t) v.data() + 63) & ~(uintptr_t) 63);
    }
};

// stage times as thread 0 sees them (each includes the wait for the slowest thread at the barrier that ends the stage)
struct stage_stats {
    double  a_us = 0.0, b_us = 0.0, c_us = 0.0, d_us = 0.0;
    int64_t n = 0;
    double  post_us = 0.0; // the post stage (logits, barrier, selection, prefetch issue), thread 0's view
    int64_t n_post = 0;
};

// the experts the last post stage chose (thread 0's copy)
struct pf_choice {
    int     n = 0;
    int32_t ids[MAX_PF] = {};
};

class team {
public:
    // `n_expert` > 0 and a multiple-of-32 `n_embd` enable the G15 post stage. `on_worker_start(ith)` runs on worker thread ith (1 .. n_threads-1) before its first job (pinning,
    // priority); `spin_us` is how long an idle worker spins before it parks
    team(int n_threads, int64_t n_embd, int64_t n_ff, int64_t n_expert = 0, std::function<void(int)> on_worker_start = nullptr, int spin_us = 4000, bool profile = false) :
        nth_(std::max(1, n_threads)), spin_us_(spin_us), profile_(profile), n_embd_(n_embd), n_expert_(n_expert), on_start_(std::move(on_worker_start)) {
        bar_.n = nth_;
        xq_.alloc((size_t) MAX_R*(4*n_embd + 64));       // up to 4 bytes per element (f32 activations)
        xf_.assign((size_t) n_embd, 0.0f);
        up_.assign((size_t) MAX_U*n_ff, 0.0f);
        gate_.assign((size_t) MAX_U*n_ff, 0.0f);
        logit_.assign((size_t) std::max<int64_t>(n_expert, 1), 0.0f);
        tc_.resize((size_t) nth_);
        for (auto & t : tc_) {
            t.act.assign((size_t) MAX_U*n_ff, 0.0f);
            t.actq.alloc((size_t) MAX_U*(4*n_ff + 64));
            t.ex.assign((size_t) MAX_U*(n_embd/nth_ + 2), 0.0f);
            if (n_expert > 0 && n_embd % 32 == 0) {
                t.xq8.alloc(ggml_row_size(GGML_TYPE_Q8_0, n_embd) + 64);
            }
        }
        arr_.resize((size_t) nth_);
        {
            // ticks of the time stamp counter per microsecond, from the steady clock
            const auto     t0 = clk::now();
            const uint64_t c0 = tsc_now(nullptr);
            while (since_us(t0) < 2000.0) { MOE_LEAN_PAUSE(); }
            const uint64_t c1 = tsc_now(nullptr);
            tpu_ = std::max(1.0, (double) (c1 - c0)/since_us(t0));
        }
        for (int i = 1; i < nth_; ++i) {
            workers_.emplace_back([this, i] { worker_main(i); });
        }
        // no job may be published before every worker is up: a worker that read job_seq_ after the first publish would wait for a job that already started
        while (ready_.load(std::memory_order_acquire) < nth_ - 1) {
            std::this_thread::yield();
        }
    }

    ~team() { stop(); }

    team(const team &) = delete;
    team & operator=(const team &) = delete;

    int n_threads() const { return nth_; }

    // joins the workers (idempotent); no run() may be in flight
    void stop() {
        if (stop_.exchange(true)) {
            return;
        }
        { std::lock_guard<std::mutex> lk(mtx_); }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers_.clear();
    }

    // thread 0: publish the job, run its share, return when every thread has passed the last barrier (y is complete). If the job asks for the post stage the caller must call post()
    // before its next run() (run() settles an unanswered post as "skip" itself, so the workers never wait for a decision that does not come).
    void run(const job & j) {
        if (pending_post_) {
            post(false);
        }
        const bool want_post = j.pf_lb != nullptr && j.pf_k > 0 && (j.pf_force != nullptr || (j.pf_rw != nullptr && j.pf_map != nullptr && n_expert_ > 0 && !tc_[0].xq8.v.empty()));
        cur_ = j;
        if (cur_.nr <= 0) {
            cur_.nr      = 1;
            cur_.rows[0] = { j.m, j.ids, j.w, j.x, j.y };
        }
        cur_.nr  = std::min(cur_.nr, MAX_R);
        cur_.n_u = 0;
        for (int r = 0; r < cur_.nr; ++r) {
            for (int e = 0; e < cur_.rows[r].m; ++e) {
                const int32_t id = cur_.rows[r].ids[e];
                int u = 0;
                while (u < cur_.n_u && !(cur_.u_id[u] == id && cur_.u_n[u] < MAX_R)) {
                    ++u;
                }
                if (u == cur_.n_u) {
                    cur_.u_id[u] = id;
                    cur_.u_n[u]  = 0;
                    cur_.n_u++;
                }
                cur_.u_pair[u][cur_.u_n[u]++] = (uint8_t) (r*MAX_M + e);
            }
        }
        if (!want_post) {
            cur_.pf_lb = nullptr;       // the workers copy this job: they must not wait for a post stage that is not coming
            cur_.pf_k  = 0;
        }
        pending_post_ = want_post;
        cur_seq_      = job_seq_.load(std::memory_order_relaxed) + 1;
        ++req_no_;
        t_pub_        = tsc_now(nullptr);
        arr_[0].t_in  = t_pub_;
        job_seq_.fetch_add(1, std::memory_order_seq_cst);
        if (parked_.load(std::memory_order_seq_cst) > 0) {
            { std::lock_guard<std::mutex> lk(mtx_); }
            cv_.notify_all();
        }
        execute(0, cur_, tc_[0]);
    }

    // thread 0, after it has answered the request: go = true runs the post stage together with the workers (they wait for this call); go = false tells them to skip it
    // (another request is already pending: it must not wait behind a prefetch). Returns true when a post stage ran.
    bool post(bool go) {
        if (!pending_post_) {
            return false;
        }
        pending_post_ = false;
        post_state_.store(((uint64_t) cur_seq_ << 2) | (go ? 1u : 2u), std::memory_order_seq_cst);
        if (go) {
            const auto t0 = profile_ ? clk::now() : clk::time_point();
            post_stage(0, tc_[0], cur_, cur_seq_);
            if (profile_) {
                stats_.post_us += since_us(t0);
                stats_.n_post++;
            }
        }
        return go;
    }

    // a request that takes longer than `us` microseconds is described by `fn` (which thread arrived last at which barrier, on which CPU, and how late each thread picked the job up)
    void set_stall_logger(double us, std::function<void(const char *)> fn) {
        stall_us_ = us;
        on_stall_ = std::move(fn);
    }

    const pf_choice & last_choice() const { return choice_; }
    int last_unique() const { return cur_.n_u; } // distinct experts of the last job
    const stage_stats & stats() const { return stats_; }

private:
    struct tctx {
        std::vector<float> act;
        abuf               actq;
        std::vector<float> ex;
        abuf               xq8;   // the request row in q8_0, for the router dot products of the post stage
        int                ls = 0;
    };

    using clk = std::chrono::steady_clock;

    static double since_us(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

    void park(uint32_t seen) {
        std::unique_lock<std::mutex> lk(mtx_);
        parked_.fetch_add(1, std::memory_order_seq_cst);
        cv_.wait(lk, [&] { return stop_.load(std::memory_order_seq_cst) || job_seq_.load(std::memory_order_seq_cst) != seen; });
        parked_.fetch_sub(1, std::memory_order_seq_cst);
    }

    void worker_main(int ith) {
        if (on_start_) {
            on_start_(ith);
        }
        uint32_t seen = 0; // job_seq_ starts at 0 and the constructor does not return before this thread has announced itself
        ready_.fetch_add(1, std::memory_order_release);
        for (;;) {
            uint32_t cur   = seen;
            uint32_t spins = 0;
            auto     t0    = clk::now();
            for (;;) {
                cur = job_seq_.load(std::memory_order_acquire);
                if (cur != seen || stop_.load(std::memory_order_relaxed)) {
                    break;
                }
                MOE_LEAN_PAUSE();
                if ((++spins & 1023) == 0 && since_us(t0) > spin_us_) {
                    park(seen);
                    t0 = clk::now();
                }
            }
            if (cur == seen) {
                return; // stop
            }
            seen = cur;
            arr_[(size_t) ith].t_in = tsc_now(nullptr);
            const job jl = cur_;                 // a private copy: the leader may publish the next job while a slow worker is still in the post stage
            execute(ith, jl, tc_[(size_t) ith]);
            if (jl.pf_lb != nullptr && jl.pf_k > 0) {
                // wait for the leader's decision on the post stage of THIS job (the tag is the job's sequence number)
                uint64_t v;
                for (;;) {
                    v = post_state_.load(std::memory_order_acquire);
                    if ((uint32_t) (v >> 2) == cur) {
                        break;
                    }
                    if (stop_.load(std::memory_order_relaxed)) {
                        return;
                    }
                    MOE_LEAN_PAUSE();
                }
                if ((v & 3) == 1) {
                    post_stage(ith, tc_[(size_t) ith], jl, cur);
                }
            }
        }
    }

    template <int HINT>
    static inline bool pf_span(const char * b, const char * e, const std::atomic<uint32_t> & seq_now, uint32_t seq, bool (*abort_fn)(void *), void * abort_ctx) {
        int cnt = 0;
        for (const char * p = (const char *) ((uintptr_t) b & ~(uintptr_t) 63); p < e; p += 64) {
            if (HINT == 4) {
                (void) *(const volatile unsigned char *) p;
            } else {
#if defined(_M_X64) || defined(__x86_64__)
                _mm_prefetch(p, HINT == 0 ? _MM_HINT_T0 : HINT == 1 ? _MM_HINT_T1 : HINT == 2 ? _MM_HINT_T2 : _MM_HINT_NTA);
#else
                __builtin_prefetch(p);
#endif
            }
            if ((++cnt & 127) == 0 && (seq_now.load(std::memory_order_relaxed) != seq || (abort_fn != nullptr && abort_fn(abort_ctx)))) {
                return false;       // the next request has been published (or is waiting): the rest of the prefetch would only get in its way
            }
        }
        return true;
    }

    // the post stage, run by every thread after the reply: logits of this thread's experts -> barrier -> the same top ranks everywhere -> prefetch this thread's slices
    void post_stage(int ith, tctx & tc, const job & j, uint32_t seq) {
        int       sel[MAX_PF];
        int       nsel = 0;
        const int K    = std::min(j.pf_k, (int) MAX_PF);
        if (j.pf_force != nullptr) {
            for (int i = 0; i < K; ++i) {
                sel[nsel++] = j.pf_force[i];
            }
        } else {
            const ggml_type_traits_cpu * t8   = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            const size_t                 rrow = ggml_row_size(GGML_TYPE_Q8_0, n_embd_);
            t8->from_float(xf_.data(), (void *) tc.xq8.p, n_embd_);
            const int64_t e0 = (ith*n_expert_)/nth_;
            const int64_t e1 = ((ith + 1)*n_expert_)/nth_;
            for (int64_t e = e0; e < e1; ++e) {
                t8->vec_dot((int) n_embd_, &logit_[(size_t) e], 0, j.pf_rw + (size_t) e*rrow, 0, tc.xq8.p, 0, 1);
            }
            bar_.wait(tc.ls);
            // top ranks, ties to the lower expert id; every thread computes the same list from the same logits
            const int pool = std::min(std::max(j.pf_pool, K), 16);
            float     tv[16];
            int       ti[16];
            int       n = 0;
            for (int64_t e = 0; e < n_expert_; ++e) {
                const float v = logit_[(size_t) e];
                if (n == pool && !(v > tv[pool - 1])) {
                    continue;
                }
                int p = n < pool ? n : pool - 1;
                while (p > 0 && v > tv[p - 1]) {
                    tv[p] = tv[p - 1];
                    ti[p] = ti[p - 1];
                    --p;
                }
                tv[p] = v;
                ti[p] = (int) e;
                if (n < pool) {
                    ++n;
                }
            }
            for (int r = 0; r < n && nsel < K; ++r) {
                if (j.pf_map[ti[r]] < 0) {
                    sel[nsel++] = ti[r];
                }
            }
        }
        if (ith == 0) {
            choice_.n = nsel;
            for (int i = 0; i < nsel; ++i) {
                choice_.ids[i] = sel[i];
            }
        }
        const layer_banks & nb    = *j.pf_lb;
        const int64_t       n_ff  = nb.up.rows;
        const int64_t       n_out = nb.down.rows;
        const int64_t       r0 = (ith*n_ff)/nth_, r1 = ((ith + 1)*n_ff)/nth_;
        const int64_t       d0 = (ith*n_out)/nth_, d1 = ((ith + 1)*n_out)/nth_;
        for (int s = 0; s < nsel; ++s) {
            const size_t e = (size_t) sel[s];
            const char * spans[3][2] = {
                { nb.up.data   + e*nb.up.expert_bytes   + (size_t) r0*nb.up.row_bytes,   nb.up.data   + e*nb.up.expert_bytes   + (size_t) r1*nb.up.row_bytes   },
                { nb.gate.data + e*nb.gate.expert_bytes + (size_t) r0*nb.gate.row_bytes, nb.gate.data + e*nb.gate.expert_bytes + (size_t) r1*nb.gate.row_bytes },
                { nb.down.data + e*nb.down.expert_bytes + (size_t) d0*nb.down.row_bytes, nb.down.data + e*nb.down.expert_bytes + (size_t) d1*nb.down.row_bytes },
            };
            for (int k = 0; k < 3; ++k) {
                bool ok;
                switch (j.pf_hint) {
                    case 0:  ok = pf_span<0>(spans[k][0], spans[k][1], job_seq_, seq, j.pf_abort, j.pf_abort_ctx); break;
                    case 2:  ok = pf_span<2>(spans[k][0], spans[k][1], job_seq_, seq, j.pf_abort, j.pf_abort_ctx); break;
                    case 3:  ok = pf_span<3>(spans[k][0], spans[k][1], job_seq_, seq, j.pf_abort, j.pf_abort_ctx); break;
                    case 4:  ok = pf_span<4>(spans[k][0], spans[k][1], job_seq_, seq, j.pf_abort, j.pf_abort_ctx); break;
                    default: ok = pf_span<1>(spans[k][0], spans[k][1], job_seq_, seq, j.pf_abort, j.pf_abort_ctx); break;
                }
                if (!ok) {
                    return;
                }
            }
        }
    }

    // leader, after the last barrier of a request: if the whole request took longer than stall_us_, say which thread was late where
    void check_stall() {
        const double total = (double) (tsc_now(nullptr) - t_pub_)/tpu_;
        if (total < stall_us_) {
            return;
        }
        char   buf[640];
        int    off = snprintf(buf, sizeof(buf), "moe_hybrid: lean request #%lld took %.1f ms (threshold %.1f ms);", (long long) req_no_, total/1e3, stall_us_/1e3);
        double pick_max = 0.0;
        int    pick_th  = 0;
        for (int i = 1; i < nth_; ++i) {
            const double d = (double) ((int64_t) (arr_[(size_t) i].t_in - t_pub_))/tpu_;
            if (d > pick_max) { pick_max = d; pick_th = i; }
        }
        off += snprintf(buf + off, sizeof(buf) - off, " job pickup: last thread %d after %.0f us (cpu at stage A %u);", pick_th, pick_max, arr_[(size_t) pick_th].cpu[0]);
        static const char * names[3] = { "A", "B", "D" };
        for (int s = 0; s < 3 && off < (int) sizeof(buf) - 120; ++s) {
            uint64_t lo = ~(uint64_t) 0, hi = 0;
            int      who = 0;
            for (int i = 0; i < nth_; ++i) {
                const uint64_t t = arr_[(size_t) i].t[s];
                if (t < lo) { lo = t; }
                if (t >= hi) { hi = t; who = i; }
            }
            off += snprintf(buf + off, sizeof(buf) - off, " stage %s: last thread %d (cpu %u) %.0f us after the first;", names[s], who, arr_[(size_t) who].cpu[s], (double) (hi - lo)/tpu_);
        }
        on_stall_(buf);
    }

    void execute(int ith, const job & j, tctx & tc) {
        const layer_banks & lb = *j.lb;

        const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(lb.up.type);
        const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(lb.gate.type);
        const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(lb.down.type);
        const ggml_type              vdt   = tu->vec_dot_type;
        const ggml_type              vdt_d = td->vec_dot_type;
        const ggml_from_float_t      q_x   = ggml_get_type_traits_cpu(vdt)->from_float;
        const ggml_from_float_t      q_a   = ggml_get_type_traits_cpu(vdt_d)->from_float;

        const int64_t k     = lb.up.k;
        const int64_t n_ff  = lb.up.rows;
        const int64_t n_out = lb.down.rows;

        const bool prof = profile_ && ith == 0;
        auto       t0   = prof ? clk::now() : clk::time_point();
        double     ta = 0.0, tb = 0.0, tcc = 0.0;

        // a (row, expert) pair p = r*MAX_M + e has its up / gate / activation / down results at index p
        const size_t xq_stride = (ggml_row_size(vdt, k) + 63) & ~(size_t) 63;

        // A. quantize the input rows; the block ranges are the ones mul_mat_id hands its threads. (With the post stage on, also keep a private f32 copy of row 0: the mailbox
        // belongs to the GPU again as soon as the reply is out.)
        {
            const int64_t bs = ggml_blck_size(vdt);
            const int64_t nb = k/bs;
            const int64_t b0 = (ith*nb)/nth_;
            const int64_t b1 = ((ith + 1)*nb)/nth_;
            if (b1 > b0) {
                for (int r = 0; r < j.nr; ++r) {
                    q_x(j.rows[r].x + b0*bs, xq_.p + (size_t) r*xq_stride + (size_t) b0*ggml_type_size(vdt), (b1 - b0)*bs);
                }
                if (j.pf_lb != nullptr && j.pf_k > 0 && (int64_t) xf_.size() == k) {
                    memcpy(xf_.data() + b0*bs, j.rows[0].x + b0*bs, (size_t) ((b1 - b0)*bs)*sizeof(float));
                }
            }
        }
        arr_[(size_t) ith].t[0] = tsc_now(&arr_[(size_t) ith].cpu[0]);
        bar_.wait(tc.ls);
        if (prof) { ta = since_us(t0); }

        // B. up and gate rows of this thread, expert by expert (two weight streams per thread); a weight row serves every token row that selected the expert
        {
            const int64_t r0 = (ith*n_ff)/nth_;
            const int64_t r1 = ((ith + 1)*n_ff)/nth_;
            for (int u = 0; u < j.n_u; ++u) {
                const char * pu = lb.up.data   + (size_t) j.u_id[u]*lb.up.expert_bytes;
                const char * pg = lb.gate.data + (size_t) j.u_id[u]*lb.gate.expert_bytes;
                const int    nu = j.u_n[u];
                for (int64_t r = r0; r < r1; ++r) {
                    for (int q = 0; q < nu; ++q) {
                        const int    p  = j.u_pair[u][q];
                        const char * xq = xq_.p + (size_t) (p/MAX_M)*xq_stride;
                        tu->vec_dot((int) k, up_.data()   + (size_t) p*n_ff + r, 0, pu + (size_t) r*lb.up.row_bytes,   0, xq, 0, 1);
                        tg->vec_dot((int) k, gate_.data() + (size_t) p*n_ff + r, 0, pg + (size_t) r*lb.gate.row_bytes, 0, xq, 0, 1);
                    }
                }
            }
        }
        arr_[(size_t) ith].t[1] = tsc_now(&arr_[(size_t) ith].cpu[1]);
        bar_.wait(tc.ls);
        if (prof) { tb = since_us(t0); }

        // C. activation and its quantization for every pair, by every thread
        const size_t arow = ggml_row_size(vdt_d, n_ff);
        for (int r = 0; r < j.nr; ++r) {
            for (int e = 0; e < j.rows[r].m; ++e) {
                const size_t p = (size_t) (r*MAX_M + e);
                ggml_cpu_swiglu_f32((int) n_ff, tc.act.data() + p*n_ff, gate_.data() + p*n_ff, up_.data() + p*n_ff);
                q_a(tc.act.data() + p*n_ff, tc.actq.p + p*arow, n_ff);
            }
        }
        if (prof) { tcc = since_us(t0); }

        // D. down rows of this thread, the router weights, the sum over each row's experts in its order
        {
            const int64_t d0 = (ith*n_out)/nth_;
            const int64_t d1 = ((ith + 1)*n_out)/nth_;
            const int64_t nd = d1 - d0;
            for (int u = 0; u < j.n_u; ++u) {
                const char * pd = lb.down.data + (size_t) j.u_id[u]*lb.down.expert_bytes;
                const int    nu = j.u_n[u];
                for (int64_t r = d0; r < d1; ++r) {
                    for (int q = 0; q < nu; ++q) {
                        const size_t p = j.u_pair[u][q];
                        td->vec_dot((int) n_ff, tc.ex.data() + p*nd + (r - d0), 0, pd + (size_t) r*lb.down.row_bytes, 0, tc.actq.p + p*arow, 0, 1);
                    }
                }
            }
            // the products first (their own loop), then the sums in expert order: the graph's ggml_mul and ggml_add are separate rounded operations
            for (int rr = 0; rr < j.nr; ++rr) {
                const row & R  = j.rows[rr];
                float *     ex = tc.ex.data() + (size_t) rr*MAX_M*nd;
                for (int e = 0; e < R.m; ++e) {
                    float *     exe = ex + (size_t) e*nd;
                    const float we  = R.w[e];
                    for (int64_t r = 0; r < nd; ++r) {
                        exe[r] = exe[r]*we;
                    }
                }
                for (int64_t r = 0; r < nd; ++r) {
                    float acc = ex[(size_t) r];
                    for (int e = 1; e < R.m; ++e) {
                        acc = acc + ex[(size_t) e*nd + (size_t) r];
                    }
                    R.y[d0 + r] = acc;
                }
            }
        }
        arr_[(size_t) ith].t[2] = tsc_now(&arr_[(size_t) ith].cpu[2]);
        bar_.wait(tc.ls);
        if (ith == 0 && on_stall_) {
            check_stall();
        }
        if (prof) {
            const double td_ = since_us(t0);
            stats_.a_us += ta;
            stats_.b_us += tb - ta;
            stats_.c_us += tcc - tb;
            stats_.d_us += td_ - tcc;
            stats_.n++;
        }
    }

    int                   nth_;
    int                   spin_us_;
    bool                  profile_;
    int64_t               n_embd_;
    int64_t               n_expert_;
    std::function<void(int)> on_start_;
    barrier               bar_;
    abuf                  xq_;
    std::vector<float>    xf_;
    std::vector<float>    up_, gate_, logit_;
    std::vector<tctx>     tc_;
    job                   cur_;
    uint32_t              cur_seq_ = 0;
    bool                  pending_post_ = false;
    pf_choice             choice_;
    stage_stats           stats_;
    struct alignas(64) arrival {
        uint64_t t[4]   = {};
        unsigned cpu[4] = {};
        uint64_t t_in   = 0;
    };
    std::vector<arrival>  arr_;
    double                tpu_       = 1.0;      // TSC ticks per microsecond
    uint64_t              t_pub_     = 0;
    int64_t               req_no_    = 0;
    double                stall_us_  = 1e18;
    std::function<void(const char *)> on_stall_;
    std::atomic<uint32_t> job_seq_{0};
    std::atomic<uint64_t> post_state_{0};
    std::atomic<int>      ready_{0};
    std::atomic<bool>     stop_{false};
    std::atomic<int>      parked_{0};
    std::mutex            mtx_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
};

} // namespace moe_lean
