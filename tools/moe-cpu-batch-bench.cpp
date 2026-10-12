// CPU time of ONE Qwen3.6-35B-A3B MoE FFN layer for an N-token batch (host-only microbenchmark: no GPU, no model file).
//
// This is the cost stock llama.cpp pays per layer in its prefill for the experts that live in system RAM (-ncmoe): ggml_mul_mat_id for up, gate and down over synthetic
// Q4_K expert banks (n_embd 2048, n_ff_exp 512, 256 experts, 8 used per token), swiglu, the router weights and the sum over the 8 experts. It is the "p(N)" of the
// short-prompt crossover study (G22): the layer-major prefill pays c = the copy of one 432 MiB bank over PCIe whatever N is, the CPU pays p(N).
//
//   moe-cpu-batch-bench [--threads 16] [--layers 6] [--type q4_K] [--ns 1,2,4,8,...] [--iters N] [--skew 0|1]
//
// Routing is uniform over the 256 experts (--skew 1: a 1/rank skew like a trained router, fewer distinct experts at small N). Banks rotate so that every call reads cold DRAM.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

struct bank { ggml_tensor *up, *gate, *down; };

static void fill_random(ggml_tensor * t, std::mt19937_64 & rng) {
    const size_t n = ggml_nbytes(t);
    uint64_t * p = (uint64_t *) t->data;
    for (size_t i = 0; i < n / 8; ++i) p[i] = rng();
    const size_t bs = ggml_type_size(t->type);
    if (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f), dm = ggml_fp32_to_fp16(0.005f);
        for (size_t o = 0; o + bs <= n; o += bs) { memcpy((char *) t->data + o, &d, 2); memcpy((char *) t->data + o + 2, &dm, 2); }
    } else if (t->type == GGML_TYPE_Q6_K) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
        for (size_t o = 0; o + bs <= n; o += bs) memcpy((char *) t->data + o + bs - 2, &d, 2);
    }
}

struct layer_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *w = nullptr, *out = nullptr;
};

static void build(layer_graph & g, const bank & b, int n_embd, int N, int k) {
    ggml_init_params ip = { ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, false), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, N);
    g.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, N);
    g.w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, N);
    ggml_set_input(g.x); ggml_set_input(g.ids); ggml_set_input(g.w);
    ggml_tensor * cur  = ggml_reshape_3d(ctx, g.x, n_embd, 1, N);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, b.up,   cur, g.ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, b.gate, cur, g.ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * ex   = ggml_mul_mat_id(ctx, b.down, act, g.ids);
    ex = ggml_mul(ctx, ex, g.w);
    ggml_tensor * out = ggml_view_2d(ctx, ex, n_embd, N, ex->nb[2], 0);
    for (int i = 1; i < k; ++i) {
        out = ggml_add(ctx, out, ggml_view_2d(ctx, ex, n_embd, N, ex->nb[2], (size_t) i * ex->nb[1]));
    }
    ggml_set_output(out);
    g.gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(g.gf, out);
    g.out = out;
    g.ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { fprintf(stderr, "gallocr failed (N=%d)\n", N); exit(1); }
}

int main(int argc, char ** argv) {
    int n_threads = 16, n_layers = 6, iters = 0, skew = 0;
    std::string tname = "q4_K";
    std::vector<int> ns = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads") n_threads = atoi(argv[++i]);
        else if (a == "--layers") n_layers = atoi(argv[++i]);
        else if (a == "--iters") iters = atoi(argv[++i]);
        else if (a == "--skew") skew = atoi(argv[++i]);
        else if (a == "--type") tname = argv[++i];
        else if (a == "--ns") {
            ns.clear();
            std::string v = argv[++i];
            size_t pos = 0;
            while (pos < v.size()) { size_t c = v.find(',', pos); if (c == std::string::npos) c = v.size(); ns.push_back(atoi(v.substr(pos, c - pos).c_str())); pos = c + 1; }
        }
    }
    ggml_type type = GGML_TYPE_Q4_K;
    if (tname == "q5_K") type = GGML_TYPE_Q5_K; else if (tname == "q6_K") type = GGML_TYPE_Q6_K;
    const int n_embd = 2048, n_ff = 512, n_exp = 256, k = 8;

    ggml_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    ggml_init_params ip = { ggml_tensor_overhead() * (3 * (size_t) n_layers + 1), nullptr, true };
    ggml_context * bctx = ggml_init(ip);
    std::vector<bank> banks(n_layers);
    for (auto & b : banks) {
        b.up   = ggml_new_tensor_3d(bctx, type, n_embd, n_ff, n_exp);
        b.gate = ggml_new_tensor_3d(bctx, type, n_embd, n_ff, n_exp);
        b.down = ggml_new_tensor_3d(bctx, type, n_ff, n_embd, n_exp);
    }
    ggml_backend_buffer_t bbuf = ggml_backend_alloc_ctx_tensors_from_buft(bctx, ggml_backend_cpu_buffer_type());
    if (!bbuf) { fprintf(stderr, "bank alloc failed\n"); return 1; }
    std::mt19937_64 rng(12345);
    for (auto & b : banks) { fill_random(b.up, rng); fill_random(b.gate, rng); fill_random(b.down, rng); }
    const double bank_mib = (double) (ggml_nbytes(banks[0].up) + ggml_nbytes(banks[0].gate) + ggml_nbytes(banks[0].down)) / 1048576.0;
    printf("type %s, one layer's expert bank %.1f MiB, %d banks rotated, %d threads (unpinned), routing %s\n", ggml_type_name(type), bank_mib, n_layers, n_threads, skew ? "skewed 1/rank" : "uniform");
    printf("%6s %12s %12s %14s %16s\n", "N", "median ms", "min ms", "us per token", "distinct experts");

    std::mt19937 r2(777);
    std::vector<double> cdf(n_exp);
    double acc = 0;
    for (int e = 0; e < n_exp; ++e) { acc += skew ? 1.0 / (e + 1) : 1.0; cdf[e] = acc; }
    for (auto & c : cdf) c /= acc;
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    for (int N : ns) {
        std::vector<layer_graph> gs(n_layers);
        for (int l = 0; l < n_layers; ++l) build(gs[l], banks[l], n_embd, N, k);
        std::vector<float> x((size_t) n_embd * N);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : x) v = nd(r2);
        std::vector<int32_t> ids((size_t) k * N);
        std::vector<float> w((size_t) k * N, 1.0f / k);
        std::vector<char> seen(n_exp, 0);
        int distinct = 0;
        for (int t = 0; t < N; ++t) {
            std::vector<int> pick;
            while ((int) pick.size() < k) {
                const double u = u01(r2);
                int e = (int) (std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
                if (e >= n_exp) e = n_exp - 1;
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int j = 0; j < k; ++j) { ids[(size_t) t * k + j] = pick[j]; if (!seen[pick[j]]) { seen[pick[j]] = 1; ++distinct; } }
        }
        for (auto & g : gs) {
            ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(g.w, w.data(), 0, w.size() * sizeof(float));
        }
        const int it = iters > 0 ? iters : std::max(n_layers * 2, std::min(60, 4096 / std::max(N, 1)));
        std::vector<double> t_ms;
        for (int j = 0; j < it + 2; ++j) {
            auto t0 = clk::now();
            ggml_backend_graph_compute(cpu, gs[j % n_layers].gf);
            const double ms = ms_since(t0);
            if (j >= 2) t_ms.push_back(ms);
        }
        std::sort(t_ms.begin(), t_ms.end());
        const double med = t_ms[t_ms.size() / 2];
        printf("%6d %12.3f %12.3f %14.1f %16d\n", N, med, t_ms.front(), med * 1000.0 / N, distinct);
        fflush(stdout);
        for (auto & g : gs) { ggml_gallocr_free(g.ga); ggml_free(g.ctx); }
    }
    return 0;
}
