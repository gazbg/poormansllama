// G14: host-only bitwise equivalence of the lean expert leg (src/llama-moe-lean.h) against the ggml graph the hybrid MoE engine used before it
// (src/llama-moe-hybrid.cpp, build_host_graph: mul_mat_id up, mul_mat_id gate, swiglu, mul_mat_id down, multiply by the router weights, sum over the experts in order).
//
// No CUDA, no model, no GPU. Random expert banks are quantized to the types a model may use; for every type pair, every number of missed experts m = 1 .. 8, every thread count
// of the lean team and a series of random requests, the lean result must equal the graph result BIT FOR BIT (memcmp), not within a tolerance: the two paths call the same
// quantizers, the same vec_dot kernels and the same swiglu routine, so any difference is a defect (a wrong slice, a missed barrier, a fused multiply-add).
//
//   test-moe-lean-expert                run the equivalence checks (what ctest runs)
//   test-moe-lean-expert --bench-pf [N] [cpus]   hot against cold: the cost of a request whose experts were prefetched into the workers' caches after the previous request
//   test-moe-lean-expert --bench [N]    time the graph against the lean team at the real shapes (Qwen3.6-35B-A3B: n_embd 2048, n_ff 512, 256 experts, Q4_K),
//                                       N requests per miss count (default 1500); informational, not a check
#include "../src/llama-moe-lean.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

static int n_checks   = 0;
static int n_failures = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        ++n_checks;                                       \
        if (!(cond)) {                                    \
            ++n_failures;                                 \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

// CPU seconds this process has used so far
static double process_cpu_seconds() {
#ifdef _WIN32
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        const auto to_s = [](const FILETIME & f) { return (double) (((uint64_t) f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7; };
        return to_s(k) + to_s(u);
    }
    return 0.0;
#else
    return (double) std::clock()/CLOCKS_PER_SEC;
#endif
}

static void pin_cpu(int cpu) {
#ifdef _WIN32
    if (cpu >= 0 && cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
    }
#else
    (void) cpu;
#endif
}

static void set_high_priority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
}

// the host graph of the engine, op for op (llama-moe-hybrid.cpp, moe_hybrid_engine::build_host_graph)
struct ref_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *  x = nullptr, * ids = nullptr, * w = nullptr, * out = nullptr;

    bool build(ggml_tensor * h_up, ggml_tensor * h_gate, ggml_tensor * h_down, int n_embd, int m) {
        ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false), nullptr, true };
        ctx = ggml_init(ip);
        x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m, 1);
        w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, m, 1);
        ggml_set_input(x);
        ggml_set_input(ids);
        ggml_set_input(w);

        ggml_tensor * cur  = ggml_reshape_3d(ctx, x, n_embd, 1, 1);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, h_up,   cur, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, h_gate, cur, ids);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * ex   = ggml_mul_mat_id(ctx, h_down, act, ids);
        ex = ggml_mul(ctx, ex, w);

        gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, ex);
        ggml_tensor * views[8];
        for (int i = 0; i < m; ++i) {
            views[i] = ggml_view_2d(ctx, ex, n_embd, 1, ex->nb[2], i*ex->nb[1]);
            ggml_build_forward_expand(gf, views[i]);
        }
        ggml_tensor * o = views[0];
        for (int i = 1; i < m; ++i) {
            o = ggml_add(ctx, o, views[i]);
            ggml_build_forward_expand(gf, o);
        }
        if (m == 1) {
            o = ggml_cont(ctx, o);
        }
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
        out = o;

        ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        return ga != nullptr && ggml_gallocr_alloc_graph(ga, gf);
    }

    ~ref_graph() {
        if (ga)  { ggml_gallocr_free(ga); }
        if (ctx) { ggml_free(ctx); }
    }
};

struct layer_weights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         up = nullptr, * gate = nullptr, * down = nullptr;

    ~layer_weights() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }
};

static bool make_layer(layer_weights & lw, ggml_backend_t cpu, ggml_type t_ud, ggml_type t_down, int n_embd, int n_ff, int n_expert, bool replicate, uint32_t seed) {
    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    lw.ctx  = ggml_init(ip);
    lw.up   = ggml_new_tensor_3d(lw.ctx, t_ud,   n_embd, n_ff,   n_expert);
    lw.gate = ggml_new_tensor_3d(lw.ctx, t_ud,   n_embd, n_ff,   n_expert);
    lw.down = ggml_new_tensor_3d(lw.ctx, t_down, n_ff,   n_embd, n_expert);
    lw.buf  = ggml_backend_alloc_ctx_tensors(lw.ctx, cpu);
    if (lw.buf == nullptr) {
        return false;
    }

    std::mt19937                          rng(seed);
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    ggml_tensor *                         ts[3] = { lw.up, lw.gate, lw.down };
    for (ggml_tensor * t : ts) {
        const int64_t k    = t->ne[0];
        const int64_t rows = t->ne[1];
        std::vector<float> src((size_t) k*rows);
        const int n_gen = replicate ? 1 : n_expert;
        for (int e = 0; e < n_gen; ++e) {
            const float scale = 1.0f/std::sqrt((float) k);
            for (auto & v : src) {
                v = nd(rng)*scale;
            }
            ggml_quantize_chunk(t->type, src.data(), (char *) t->data + (size_t) e*t->nb[2], 0, rows, k, nullptr);
        }
        for (int e = n_gen; e < n_expert; ++e) {
            memcpy((char *) t->data + (size_t) e*t->nb[2], t->data, t->nb[2]);
        }
    }
    return true;
}

static moe_lean::layer_banks banks_of(const layer_weights & lw) {
    moe_lean::layer_banks lb;
    lb.up   = moe_lean::bank_of(lw.up);
    lb.gate = moe_lean::bank_of(lw.gate);
    lb.down = moe_lean::bank_of(lw.down);
    return lb;
}

struct type_pair {
    ggml_type   ud, down;
    const char * name;
};

static void check_pair(ggml_backend_t cpu, const type_pair & tp, int n_embd, int n_ff, int n_expert) {
    if (ggml_quantize_requires_imatrix(tp.ud) || ggml_quantize_requires_imatrix(tp.down)) {
        std::printf("skip %s (needs an importance matrix)\n", tp.name);
        return;
    }
    std::printf("%s: quantizing the banks ...\n", tp.name);
    layer_weights lw;
    if (!make_layer(lw, cpu, tp.ud, tp.down, n_embd, n_ff, n_expert, false, 12345u)) {
        CHECK(false, "%s: weight allocation failed", tp.name);
        return;
    }
    std::printf("%s: banks ready, comparing ...\n", tp.name);
    const moe_lean::layer_banks lb = banks_of(lw);
    if (!moe_lean::supported(lb)) {
        std::printf("note %s: the lean leg does not take this type pair (the engine keeps the graph for it)\n", tp.name);
        return;
    }

    ggml_backend_cpu_set_n_threads(cpu, 4);

    std::vector<ref_graph> graphs(9);
    for (int m = 1; m <= 8; ++m) {
        CHECK(graphs[(size_t) m].build(lw.up, lw.gate, lw.down, n_embd, m), "%s: graph m=%d allocation", tp.name, m);
    }

    std::mt19937                    rng(777u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.02f, 0.6f);

    const int thread_counts[] = { 1, 2, 3, 5, 8 };
    int       bad = 0, total = 0;
    for (int nth : thread_counts) {
        moe_lean::team team(nth, n_embd, n_ff, 0, nullptr, 2000);
        for (int m = 1; m <= 8; ++m) {
            for (int rep = 0; rep < 6; ++rep) {
                std::vector<float>   x((size_t) n_embd);
                std::vector<int32_t> pool((size_t) n_expert);
                float                w[8];
                for (auto & v : x) { v = nd(rng); }
                for (int i = 0; i < n_expert; ++i) { pool[(size_t) i] = i; }
                std::shuffle(pool.begin(), pool.end(), rng);
                for (int i = 0; i < m; ++i) { w[i] = ud(rng); }

                ref_graph & g = graphs[(size_t) m];
                memcpy(g.x->data,   x.data(),    (size_t) n_embd*sizeof(float));
                memcpy(g.ids->data, pool.data(), (size_t) m*sizeof(int32_t));
                memcpy(g.w->data,   w,           (size_t) m*sizeof(float));
                const ggml_status st = ggml_backend_graph_compute(cpu, g.gf);
                CHECK(st == GGML_STATUS_SUCCESS, "%s: graph compute", tp.name);

                std::vector<float> y((size_t) n_embd, -1.0f);
                moe_lean::job j;
                j.lb = &lb; j.m = m; j.ids = pool.data(); j.w = w; j.x = x.data(); j.y = y.data();
                team.run(j);

                ++total;
                if (memcmp(y.data(), g.out->data, (size_t) n_embd*sizeof(float)) != 0) {
                    ++bad;
                    int first = -1, nd_ = 0;
                    double maxd = 0.0;
                    const float * ref = (const float *) g.out->data;
                    for (int i = 0; i < n_embd; ++i) {
                        if (memcmp(&y[(size_t) i], &ref[i], sizeof(float)) != 0) {
                            if (first < 0) { first = i; }
                            ++nd_;
                            maxd = std::max(maxd, (double) std::fabs(y[(size_t) i] - ref[i]));
                        }
                    }
                    if (bad <= 3) {
                        std::printf("  %s nth=%d m=%d rep=%d: %d of %d floats differ, first at %d (lean %.9g, graph %.9g), max |diff| %.3g\n",
                                    tp.name, nth, m, rep, nd_, n_embd, first, (double) y[(size_t) first], (double) ref[first], maxd);
                    }
                }
            }
        }
    }
    CHECK(bad == 0, "%s: %d of %d requests differ from the graph", tp.name, bad, total);
    std::printf("%-22s %4d requests x (thread counts 1,2,3,5,8) %s\n", tp.name, total, bad == 0 ? "bit-identical" : "DIFFER");
}

// several token rows in one job (the rows of an MTP verify step): every row must get exactly the bits a job of that row alone gives, with experts shared between rows or not
static void check_multi_row(ggml_backend_t cpu) {
    const int n_embd = 2048, n_ff = 512, n_expert = 12;
    const type_pair pairs[] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, "q4_K/q5_K" },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, "q4_K/q6_K" },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "q8_0/q8_0" },
    };
    for (const type_pair & tp : pairs) {
        layer_weights lw;
        if (!make_layer(lw, cpu, tp.ud, tp.down, n_embd, n_ff, n_expert, false, 4242u)) {
            CHECK(false, "%s: weight allocation failed", tp.name);
            continue;
        }
        const moe_lean::layer_banks lb = banks_of(lw);
        std::mt19937                          rng(99u);
        std::normal_distribution<float>       nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ud(0.02f, 0.6f);
        int bad = 0, total = 0, shared = 0;
        for (int nth : { 1, 3, 4, 8 }) {
            moe_lean::team team(nth, n_embd, n_ff, 0, nullptr, 2000);
            for (int rep = 0; rep < 24; ++rep) {
                const int nr = 2 + rep % 3;
                std::vector<std::vector<float>> x((size_t) nr), y1((size_t) nr), yb((size_t) nr);
                int32_t ids[moe_lean::MAX_R][8];
                float   w[moe_lean::MAX_R][8];
                int     m[moe_lean::MAX_R];
                int     pairs_n = 0;
                for (int r = 0; r < nr; ++r) {
                    x[(size_t) r].resize((size_t) n_embd);
                    for (auto & v : x[(size_t) r]) { v = nd(rng); }
                    y1[(size_t) r].assign((size_t) n_embd, -1.0f);
                    yb[(size_t) r].assign((size_t) n_embd, -2.0f);
                    std::vector<int32_t> pool((size_t) n_expert);
                    for (int i = 0; i < n_expert; ++i) { pool[(size_t) i] = i; }
                    std::shuffle(pool.begin(), pool.end(), rng);
                    m[r] = 1 + (int) (rng() % 8);
                    for (int i = 0; i < m[r]; ++i) {
                        ids[r][i] = pool[(size_t) i];
                        w[r][i]   = ud(rng);
                    }
                    pairs_n += m[r];
                }
                for (int r = 0; r < nr; ++r) {
                    moe_lean::job j;
                    j.lb = &lb; j.m = m[r]; j.ids = ids[r]; j.w = w[r]; j.x = x[(size_t) r].data(); j.y = y1[(size_t) r].data();
                    team.run(j);
                }
                moe_lean::job jb;
                jb.lb = &lb;
                jb.nr = nr;
                for (int r = 0; r < nr; ++r) {
                    jb.rows[r] = { m[r], ids[r], w[r], x[(size_t) r].data(), yb[(size_t) r].data() };
                }
                team.run(jb);
                CHECK(team.last_unique() <= pairs_n, "%s: %d distinct experts for %d pairs", tp.name, team.last_unique(), pairs_n);
                shared += pairs_n - team.last_unique();
                for (int r = 0; r < nr; ++r) {
                    ++total;
                    if (memcmp(y1[(size_t) r].data(), yb[(size_t) r].data(), (size_t) n_embd*sizeof(float)) != 0) {
                        ++bad;
                    }
                }
            }
        }
        CHECK(bad == 0, "%s: %d of %d rows of multi-row jobs differ from single-row jobs", tp.name, bad, total);
        CHECK(shared > 0, "%s: the random rows never shared an expert (the test covers nothing)", tp.name);
        std::printf("%-22s %4d rows in multi-row jobs (%d shared pairs) %s\n", tp.name, total, shared, bad == 0 ? "bit-identical to single-row jobs" : "DIFFER");
    }
}

// G15: the post stage after the reply (router logits of the next layer on the request row, one barrier, the first K non-resident experts of the top ranks, prefetch of the
// workers' slices). It must change no result bit, must choose exactly the experts a single-threaded reference chooses, and must not leave a worker waiting when the leader skips it
// or forgets it.
static void check_post_stage(ggml_backend_t cpu) {
    const int n_embd = 512, n_ff = 256, n_expert = 64;
    layer_weights cur, nxt;
    if (!make_layer(cur, cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert, false, 21u) ||
        !make_layer(nxt, cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert, false, 22u)) {
        CHECK(false, "post stage: weight allocation failed");
        return;
    }
    const moe_lean::layer_banks lb  = banks_of(cur);
    const moe_lean::layer_banks lbn = banks_of(nxt);

    std::mt19937                    rng(99u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              router((size_t) n_expert*n_embd);
    for (auto & v : router) { v = nd(rng)/std::sqrt((float) n_embd); }
    const std::vector<char> rq = moe_lean::quantize_router_q8(router.data(), n_expert, n_embd);

    const ggml_type_traits_cpu * t8   = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const size_t                 rrow = ggml_row_size(GGML_TYPE_Q8_0, n_embd);

    int bad = 0, total = 0, chosen_total = 0;
    for (int nth : { 1, 2, 3, 5, 8 }) {
        moe_lean::team team(nth, n_embd, n_ff, n_expert, nullptr, 2000);
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<float>   x((size_t) n_embd), y0((size_t) n_embd), y1((size_t) n_embd);
            for (auto & v : x) { v = nd(rng); }
            std::vector<int32_t> pool((size_t) n_expert);
            for (int i = 0; i < n_expert; ++i) { pool[(size_t) i] = i; }
            std::shuffle(pool.begin(), pool.end(), rng);
            const int            m = 1 + trial % 4;
            float                w[8];
            for (int i = 0; i < 8; ++i) { w[i] = 0.1f + 0.05f*(float) i; }
            std::vector<int32_t> map((size_t) n_expert);
            for (int e = 0; e < n_expert; ++e) { map[(size_t) e] = (rng() % 100) < 40 ? (int32_t) (rng() % 16) : -1; }
            const int K    = 1 + trial % 3;
            const int pool_n = 4 + (trial % 3)*4;                    // 4, 8, 12

            // the result without a post stage
            moe_lean::job j0;
            j0.lb = &lb; j0.m = m; j0.ids = pool.data(); j0.w = w; j0.x = x.data(); j0.y = y0.data();
            team.run(j0);

            // the same request with the post stage on
            moe_lean::job j1 = j0;
            j1.y = y1.data(); j1.pf_lb = &lbn; j1.pf_rw = rq.data(); j1.pf_map = map.data(); j1.pf_k = K; j1.pf_pool = pool_n; j1.pf_hint = trial % 4;
            team.run(j1);
            const int mode = trial % 4;                              // 0,1,2: post(true); 3: the caller skips and the next run() settles it
            if (mode == 3) {
                // no post() call at all; the next run() must settle it as "skip"
            } else {
                team.post(mode != 2);
            }
            ++total;
            if (memcmp(y0.data(), y1.data(), (size_t) n_embd*sizeof(float)) != 0) { ++bad; }

            if (mode == 0 || mode == 1) {
                // single-threaded reference of the choice: the same quantized row and dot products, top ranks, ties to the lower id, then the first K non-resident
                std::vector<char>  xq(rrow + 64);
                t8->from_float(x.data(), xq.data(), n_embd);
                std::vector<float> lg((size_t) n_expert);
                for (int e = 0; e < n_expert; ++e) {
                    t8->vec_dot(n_embd, &lg[(size_t) e], 0, rq.data() + (size_t) e*rrow, 0, xq.data(), 0, 1);
                }
                std::vector<int> ord((size_t) n_expert);
                for (int e = 0; e < n_expert; ++e) { ord[(size_t) e] = e; }
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
                std::vector<int> want;
                for (int r = 0; r < pool_n && (int) want.size() < K; ++r) {
                    if (map[(size_t) ord[(size_t) r]] < 0) { want.push_back(ord[(size_t) r]); }
                }
                const moe_lean::pf_choice & c = team.last_choice();
                bool same = c.n == (int) want.size();
                for (int i = 0; same && i < c.n; ++i) { same = c.ids[i] == want[(size_t) i]; }
                ++total;
                chosen_total += c.n;
                if (!same) {
                    ++bad;
                    if (bad <= 3) {
                        std::printf("  post stage nth=%d trial=%d: chose %d experts (", nth, trial, c.n);
                        for (int i = 0; i < c.n; ++i) { std::printf("%d ", c.ids[i]); }
                        std::printf(") want %zu (", want.size());
                        for (int e : want) { std::printf("%d ", e); }
                        std::printf(")\n");
                    }
                }
            }
        }
        // a forced choice (the test hook the benchmark uses) and a final plain request after the last post
        moe_lean::job jf;
        std::vector<float> x((size_t) n_embd, 0.5f), y((size_t) n_embd);
        const int32_t ids[2] = { 3, 7 };
        const float   w[2]   = { 0.6f, 0.4f };
        jf.lb = &lb; jf.m = 2; jf.ids = ids; jf.w = w; jf.x = x.data(); jf.y = y.data();
        jf.pf_lb = &lbn; jf.pf_force = ids; jf.pf_k = 2;
        team.run(jf);
        team.post(true);
        ++total;
        CHECK(team.last_choice().n == 2 && team.last_choice().ids[0] == 3 && team.last_choice().ids[1] == 7, "forced prefetch choice");
        team.run(jf);   // post left unanswered on purpose: run() settles it
        jf.pf_lb = nullptr; jf.pf_k = 0;
        team.run(jf);
    }
    CHECK(bad == 0, "post stage: %d of %d checks failed", bad, total);
    std::printf("post stage             %4d checks (results unchanged, choices equal to the reference, skip / forget settle), %d experts chosen: %s\n",
                total, chosen_total, bad == 0 ? "ok" : "FAIL");
}

// hot against cold at the real shapes. Request A (one expert of layer a) is followed by request B (two experts of layer b) after a gap, as in the engine; B's experts were
// prefetched into the workers' caches right after A's reply (hot), or other experts of layer b were (wrong), or nothing was (cold). The time of B is what is compared.
static void bench_pf(ggml_backend_t cpu, int reqs, std::vector<int> cpus) {
    const int n_embd = 2048, n_ff = 512, n_expert = 256, n_layer = 6;
    if (cpus.empty()) {
        for (int i = 0; i < 8; ++i) { cpus.push_back(2*i); }
    }
    const int nth = (int) cpus.size();
    std::vector<layer_weights>         lws((size_t) n_layer);
    std::vector<moe_lean::layer_banks> lbs((size_t) n_layer);
    std::printf("bench-pf: building %d layers x 3 x %.0f MiB (Q4_K, 256 experts) ...\n", n_layer, 144.0);
    for (int l = 0; l < n_layer; ++l) {
        if (!make_layer(lws[(size_t) l], cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert, true, 99u + (uint32_t) l)) {
            std::printf("allocation failed\n");
            return;
        }
        lbs[(size_t) l] = banks_of(lws[(size_t) l]);
    }
    std::mt19937                    rng(777u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              x((size_t) n_embd), y((size_t) n_embd);
    for (auto & v : x) { v = nd(rng); }
    float w[8] = { 0.3f, 0.2f, 0.15f, 0.1f, 0.08f, 0.07f, 0.05f, 0.05f };

    set_high_priority();
    pin_cpu(cpus.back());
    moe_lean::team team(nth, n_embd, n_ff, n_expert, [&](int ith) { pin_cpu(cpus[(size_t) ith - 1]); set_high_priority(); }, 20000, true);

    using clk = std::chrono::steady_clock;
    auto spin_us = [](double us) {
        const auto t = clk::now();
        while (std::chrono::duration<double, std::micro>(clk::now() - t).count() < us) { MOE_LEAN_PAUSE(); }
    };
    auto stat = [](std::vector<double> & v, double & mean, double & p50) {
        std::sort(v.begin(), v.end());
        double s = 0.0;
        for (double d : v) { s += d; }
        mean = s/(double) v.size();
        p50  = v[v.size()/2];
    };

    std::printf("\nB = 2 experts of layer b after A = 1 expert of layer a, gap between A's reply and B in the table; us per request B, %d trials per cell, %d threads on CPUs", reqs, nth);
    for (int c : cpus) { std::printf(" %d", c); }
    std::printf("\n gap  hint | cold mean  p50 | hot(K=2) mean  p50  saved | wrong mean  p50  | post stage us\n");
    for (int hint : { 4, 1, 0, 2, 3 }) {
        for (double gap : { 40.0, 70.0, 100.0, 150.0 }) {
            if (hint != 1 && hint != 4 && gap != 100.0) { continue; }
            std::vector<double> t_cold, t_hot, t_wrong, t_rep;
            double post_us0 = team.stats().post_us;
            int64_t post_n0 = team.stats().n_post;
            for (int i = 0; i < reqs*4 + 60; ++i) {
                const int mode = i % 4;                                  // 0 cold, 1 hot, 2 wrong, 3 repeat (the same request B twice in a row: the second is the ideal hot case)
                const int la = (int) (rng() % (uint32_t) n_layer);
                int lb_ = (int) (rng() % (uint32_t) n_layer);
                if (lb_ == la) { lb_ = (lb_ + 1) % n_layer; }
                int32_t ea[1] = { (int32_t) (rng() % (uint32_t) n_expert) };
                int32_t eb[2], ew[2];
                eb[0] = (int32_t) (rng() % (uint32_t) n_expert);
                do { eb[1] = (int32_t) (rng() % (uint32_t) n_expert); } while (eb[1] == eb[0]);
                do { ew[0] = (int32_t) (rng() % (uint32_t) n_expert); } while (ew[0] == eb[0] || ew[0] == eb[1]);
                do { ew[1] = (int32_t) (rng() % (uint32_t) n_expert); } while (ew[1] == ew[0] || ew[1] == eb[0] || ew[1] == eb[1]);
                moe_lean::job ja;
                ja.lb = &lbs[(size_t) la]; ja.m = 1; ja.ids = ea; ja.w = w; ja.x = x.data(); ja.y = y.data();
                if (mode == 1 || mode == 2) {
                    ja.pf_lb = &lbs[(size_t) lb_]; ja.pf_k = 2; ja.pf_hint = hint; ja.pf_force = (mode == 1) ? eb : ew;
                }
                team.run(ja);
                team.post(mode == 1 || mode == 2);
                spin_us(gap);
                moe_lean::job jb;
                jb.lb = &lbs[(size_t) lb_]; jb.m = 2; jb.ids = eb; jb.w = w; jb.x = x.data(); jb.y = y.data();
                if (mode == 3) {
                    team.run(jb);                                        // the first read brings the experts into the caches
                    spin_us(gap);
                }
                const auto t0 = clk::now();
                team.run(jb);
                const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
                if (i >= 60) { (mode == 0 ? t_cold : mode == 1 ? t_hot : mode == 2 ? t_wrong : t_rep).push_back(us); }
                spin_us(30.0);
            }
            double mc, pc, mh, ph, mw, pw, mr, pr;
            stat(t_cold, mc, pc); stat(t_hot, mh, ph); stat(t_wrong, mw, pw); stat(t_rep, mr, pr);
            const double post_avg = (team.stats().post_us - post_us0)/(double) std::max<int64_t>(1, team.stats().n_post - post_n0);
            std::printf(" %4.0f  %d    | %9.1f %6.1f | %9.1f %6.1f %6.1f | %9.1f %6.1f | %6.1f | repeat (ideal hot) %6.1f %6.1f\n", gap, hint, mc, pc, mh, ph, mc - mh, mw, pw, post_avg, mr, pr);
        }
    }
    std::printf("(saved = cold mean - hot mean for a request of 2 experts; per expert divide by 2. 'wrong' = two other experts of the layer were prefetched.)\n");
}

// micro-benchmark at the real shapes: one request = m random experts of one of n_layer layers (a working set far above the caches, so the weights come from DRAM)
static void bench(ggml_backend_t cpu, int reqs, std::vector<int> cpus) {
    const int n_embd = 2048, n_ff = 512, n_expert = 256, n_layer = 6;
    if (cpus.empty()) {
        for (int i = 0; i < 8; ++i) { cpus.push_back(2*i); }
    }
    const int nth = (int) cpus.size();
    std::vector<layer_weights> lws((size_t) n_layer);
    std::vector<moe_lean::layer_banks> lbs((size_t) n_layer);
    std::printf("bench: building %d layers x 3 x %.0f MiB (Q4_K, 256 experts) ...\n", n_layer, 144.0);
    for (int l = 0; l < n_layer; ++l) {
        if (!make_layer(lws[(size_t) l], cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert, true, 99u + (uint32_t) l)) {
            std::printf("bench: allocation failed\n");
            return;
        }
        lbs[(size_t) l] = banks_of(lws[(size_t) l]);
    }

    // graph path, set up as the engine does it: its own pinned, polling thread pool created on the calling thread
    ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
    memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
    for (int c : cpus) {
        tpp.cpumask[c] = true;
    }
    tpp.strict_cpu = true;
    tpp.poll       = 100;
    tpp.prio       = GGML_SCHED_PRIO_HIGH;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(cpu, tp);
    ggml_backend_cpu_set_n_threads(cpu, nth);

    std::vector<std::vector<ref_graph>> graphs((size_t) n_layer, std::vector<ref_graph>(9));
    for (int l = 0; l < n_layer; ++l) {
        for (int m = 1; m <= 8; ++m) {
            graphs[(size_t) l][(size_t) m].build(lws[(size_t) l].up, lws[(size_t) l].gate, lws[(size_t) l].down, n_embd, m);
        }
    }

    std::mt19937                    rng(4242u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              x((size_t) n_embd), y((size_t) n_embd);
    for (auto & v : x) { v = nd(rng); }
    std::vector<int32_t> pool((size_t) n_expert);
    for (int i = 0; i < n_expert; ++i) { pool[(size_t) i] = i; }
    float w[8] = { 0.3f, 0.2f, 0.15f, 0.1f, 0.08f, 0.07f, 0.05f, 0.05f };

    using clk = std::chrono::steady_clock;
    struct res { double mean, p50, p99; };
    auto stat = [](std::vector<double> & v) {
        std::sort(v.begin(), v.end());
        double s = 0.0;
        for (double d : v) { s += d; }
        return res{ s/(double) v.size(), v[v.size()/2], v[(size_t) ((double) v.size()*0.99)] };
    };

    std::vector<res> r_graph(9), r_lean(9);
    {
        // the graph path: the calling thread is worker 0 of the pool (pinned by ggml_threadpool_new to the last CPU of the mask)
        for (int m = 1; m <= 8; ++m) {
            std::vector<double> t;
            for (int i = 0; i < reqs + 100; ++i) {
                const int l = (int) (rng() % (uint32_t) n_layer);
                std::shuffle(pool.begin(), pool.end(), rng);
                ref_graph & g = graphs[(size_t) l][(size_t) m];
                memcpy(g.x->data, x.data(), (size_t) n_embd*sizeof(float));
                memcpy(g.ids->data, pool.data(), (size_t) m*sizeof(int32_t));
                memcpy(g.w->data, w, (size_t) m*sizeof(float));
                const auto t0 = clk::now();
                ggml_backend_graph_compute(cpu, g.gf);
                const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
                if (i >= 100) { t.push_back(us); }
                // a short idle gap like the engine sees between two requests
                const auto tg = clk::now();
                while (std::chrono::duration<double, std::micro>(clk::now() - tg).count() < 40.0) { MOE_LEAN_PAUSE(); }
            }
            r_graph[(size_t) m] = stat(t);
        }
    }
    ggml_backend_cpu_set_threadpool(cpu, nullptr);
    ggml_threadpool_free(tp);
    graphs.clear();

    {
        set_high_priority();
        pin_cpu(cpus.back());
        moe_lean::team team(nth, n_embd, n_ff, 0, [&](int ith) { pin_cpu(cpus[(size_t) ith - 1]); set_high_priority(); }, 4000, true);
        for (int m = 1; m <= 8; ++m) {
            std::vector<double> t;
            for (int i = 0; i < reqs + 100; ++i) {
                const int l = (int) (rng() % (uint32_t) n_layer);
                std::shuffle(pool.begin(), pool.end(), rng);
                moe_lean::job j;
                j.lb = &lbs[(size_t) l]; j.m = m; j.ids = pool.data(); j.w = w; j.x = x.data(); j.y = y.data();
                const auto t0 = clk::now();
                team.run(j);
                const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
                if (i >= 100) { t.push_back(us); }
                const auto tg = clk::now();
                while (std::chrono::duration<double, std::micro>(clk::now() - tg).count() < 40.0) { MOE_LEAN_PAUSE(); }
            }
            r_lean[(size_t) m] = stat(t);
        }
        const moe_lean::stage_stats & s = team.stats();
        std::printf("lean stages as thread 0 sees them (mean per request over all m): A %.2f  B %.2f  C %.2f  D %.2f us (n=%lld)\n",
                    s.a_us/(double) s.n, s.b_us/(double) s.n, s.c_us/(double) s.n, s.d_us/(double) s.n, (long long) s.n);
    }

    std::string cpu_list;
    for (int c : cpus) { cpu_list += (cpu_list.empty() ? "" : ",") + std::to_string(c); }
    std::printf("\n m | graph mean  p50   p99 | lean mean  p50   p99 | saved (mean)   [us per request, %d requests per m, %d threads on CPUs %s]\n", reqs, nth, cpu_list.c_str());
    for (int m = 1; m <= 8; ++m) {
        const res & a = r_graph[(size_t) m];
        const res & b = r_lean[(size_t) m];
        std::printf(" %d | %9.1f %6.1f %6.1f | %8.1f %6.1f %6.1f | %6.1f\n", m, a.mean, a.p50, a.p99, b.mean, b.p50, b.p99, a.mean - b.mean);
    }
    // linear fit mean = fixed + slope*m
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<res> & r = pass == 0 ? r_graph : r_lean;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int m = 1; m <= 8; ++m) { sx += m; sy += r[(size_t) m].mean; sxx += (double) m*m; sxy += m*r[(size_t) m].mean; }
        const double slope = (8*sxy - sx*sy)/(8*sxx - sx*sx);
        std::printf("%s: fixed %.1f us + %.1f us per expert\n", pass == 0 ? "graph" : "lean ", (sy - slope*sx)/8, slope);
    }
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        std::printf("no CPU backend\n");
        return 1;
    }

    if (argc > 1 && (std::string(argv[1]) == "--bench-pf" || std::string(argv[1]) == "--bench")) {
        const bool pf = std::string(argv[1]) == "--bench-pf";
        std::vector<int> cpus; // optional: comma separated logical CPUs, the last one is the calling thread's
        if (argc > 3) {
            std::string s = argv[3];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t nx = s.find(',', pos);
                if (nx == std::string::npos) { nx = s.size(); }
                cpus.push_back(std::atoi(s.substr(pos, nx - pos).c_str()));
                pos = nx + 1;
            }
        }
        if (pf) {
            bench_pf(cpu, argc > 2 ? std::atoi(argv[2]) : 400, cpus);
        } else {
            bench(cpu, argc > 2 ? std::atoi(argv[2]) : 1500, cpus);
        }
        ggml_backend_free(cpu);
        return 0;
    }

    const int n_embd = 2048, n_ff = 512, n_expert = 10;
    const type_pair pairs[] = {
        { GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K,   "q4_K/q4_K"      },
        { GGML_TYPE_Q4_K,   GGML_TYPE_Q6_K,   "q4_K/q6_K"      },
        { GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K,   "q5_K/q5_K"      },
        { GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,   "q8_0/q8_0"      },
        { GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0,   "q4_0/q4_0"      },
        { GGML_TYPE_Q5_1,   GGML_TYPE_Q5_0,   "q5_1/q5_0"      },
        { GGML_TYPE_Q2_K,   GGML_TYPE_Q3_K,   "q2_K/q3_K"      },
        { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "iq4_xs/iq4_nl"  },
        { GGML_TYPE_MXFP4,  GGML_TYPE_MXFP4,  "mxfp4/mxfp4"    },
        { GGML_TYPE_F16,    GGML_TYPE_F16,    "f16/f16"        },
        { GGML_TYPE_BF16,   GGML_TYPE_BF16,   "bf16/bf16"      },
        { GGML_TYPE_F32,    GGML_TYPE_F32,    "f32/f32"        },
    };
    for (const type_pair & tp : pairs) {
        check_pair(cpu, tp, n_embd, n_ff, n_expert);
    }
    check_multi_row(cpu);

    // an idle team parks: after the last request the workers spin for the idle window (4 ms here) and then sleep, so a generation that has ended does not leave 8 cores spinning;
    // the first request after a park wakes them and still gives the same bits
    {
        layer_weights lw;
        if (make_layer(lw, cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 512, 256, 6, false, 9u)) {
            const moe_lean::layer_banks lb = banks_of(lw);
            moe_lean::team team(8, 512, 256, 0, nullptr, 4000);
            std::mt19937 rng(31u);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> x(512), y0(512), y1(512);
            for (auto & v : x) { v = nd(rng); }
            const int32_t ids[3] = { 4, 1, 5 };
            const float   w[3]   = { 0.5f, 0.3f, 0.2f };
            moe_lean::job j;
            j.lb = &lb; j.m = 3; j.ids = ids; j.w = w; j.x = x.data(); j.y = y0.data();
            for (int i = 0; i < 20; ++i) { team.run(j); }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));      // let the workers reach their park
            const double c0 = process_cpu_seconds();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            const double used = process_cpu_seconds() - c0;
            CHECK(used < 0.5, "an idle 8-thread team used %.3f CPU-s in 1 s (workers did not park)", used);
            std::printf("idle team: %.3f CPU-s over 1 s (8 threads, parked)\n", used);
            // the stall diagnostic: with a zero threshold every request is described; the text names the stages and a CPU
            int         n_stall = 0;
            std::string last_stall;
            team.set_stall_logger(0.0, [&](const char * s) { ++n_stall; last_stall = s; });
            j.y = y1.data();
            const auto t0 = std::chrono::steady_clock::now();
            team.run(j);
            const double wake_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            std::printf("first request after the park: %.0f us\n", wake_us);
            CHECK(memcmp(y0.data(), y1.data(), 512*sizeof(float)) == 0, "the request after a park gave different bits");
            CHECK(n_stall == 1 && last_stall.find("stage A") != std::string::npos && last_stall.find("stage D") != std::string::npos && last_stall.find("job pickup") != std::string::npos,
                  "the stall logger did not describe the request (%d calls): %s", n_stall, last_stall.c_str());
            std::printf("stall logger sample: %s\n", last_stall.c_str());
        }
    }

    check_post_stage(cpu);

    // layers the lean leg must refuse (the engine then keeps the graph for them)
    {
        layer_weights lw;
        if (make_layer(lw, cpu, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 512, 256, 4, false, 5u)) {
            moe_lean::layer_banks lb = banks_of(lw);
            CHECK(moe_lean::supported(lb), "a plain q4_K layer must be taken");
            moe_lean::layer_banks bad = lb;
            bad.gate.type = GGML_TYPE_Q5_K;                 // up and gate of different types
            CHECK(!moe_lean::supported(bad), "up and gate of different types must be refused");
            bad = lb;
            bad.down.k = lb.down.k + 1;                     // down input width not the up output width
            CHECK(!moe_lean::supported(bad), "a down bank that does not match the up bank must be refused");
            bad = lb;
            bad.up.k = 300; bad.gate.k = 300;               // not a multiple of the q8_K block
            CHECK(!moe_lean::supported(bad), "a row length that is not a block multiple must be refused");
        }
    }

    std::printf("test-moe-lean-expert: %d checks, %d failures\n", n_checks, n_failures);
    ggml_backend_free(cpu);
    return n_failures == 0 ? 0 : 1;
}
