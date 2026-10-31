// Latency of serving the missed experts of ONE layer on a second GPU (the extra expert-cache tier, plan H74).
// One request = copy the activation x (2048 floats), the expert slots (m ints) and the router weights (m floats) to the device, run up / gate / swiglu / down /
// weighted sum of m experts through the MUL_MAT_ID kernels of the CUDA backend, copy the 2048 result floats back. Expert dimensions of Qwen3.6-35B-A3B: n_embd 2048, expert FFN 512,
// gate / up Q4_K, down Q5_K (Q4_K_M). Weights are zeros (the kernels have no data-dependent time). Timing is wall clock around the whole request, so it includes launch and synchronisation.
//   g++ -O2 -I third_party/llama.cpp/ggml/include tools/pascal/gpu1_serve_bench.cpp -L build/llama-cuda/bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
//       -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/gpu1_serve_bench
//   build/pascal/gpu1_serve_bench [device=1] [slots=110] [iterations=3000]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct req_graph {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph *  gf    = nullptr;
    ggml_tensor *  x = nullptr, * ids = nullptr, * w = nullptr, * out = nullptr;
};

static req_graph build(ggml_backend_t be, ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w, int m, int n_embd, int n_ff) {
    req_graph g;
    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.x   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, n_embd, 1, 1);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, m, 1);
    g.w   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, m, 1);
    ggml_tensor * u = ggml_mul_mat_id(g.ctx, up_w,   g.x, g.ids);   // [n_ff, m, 1]
    ggml_tensor * q = ggml_mul_mat_id(g.ctx, gate_w, g.x, g.ids);
    ggml_tensor * a = ggml_swiglu_split(g.ctx, q, u);
    ggml_tensor * d = ggml_mul_mat_id(g.ctx, down_w, a, g.ids);     // [n_embd, m, 1]
    d = ggml_mul(g.ctx, d, g.w);
    ggml_tensor * p = ggml_cont(g.ctx, ggml_permute(g.ctx, d, 1, 0, 2, 3)); // [m, n_embd, 1]
    g.out = ggml_sum_rows(g.ctx, p);                                // [1, n_embd, 1]
    (void) n_ff;
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { std::fprintf(stderr, "alloc failed\n"); std::exit(1); }
    return g;
}

int main(int argc, char ** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 1;
    const int S      = argc > 2 ? std::atoi(argv[2]) : 110;
    const int iters  = argc > 3 ? std::atoi(argv[3]) : 3000;
    const int n_embd = 2048, n_ff = 512;

    ggml_backend_t be = ggml_backend_cuda_init(device);
    if (!be) { std::fprintf(stderr, "cannot init CUDA device %d\n", device); return 1; }

    // the cache tier: S slots of gate / up (Q4_K) and down (Q5_K) of one layer (a real tier has 40 layers of this)
    ggml_init_params wp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * gate_w = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, n_embd, n_ff, S);
    ggml_tensor * up_w   = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, n_embd, n_ff, S);
    ggml_tensor * down_w = ggml_new_tensor_3d(wctx, GGML_TYPE_Q5_K, n_ff, n_embd, S);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    for (ggml_tensor * t : { gate_w, up_w, down_w }) {
        std::vector<char> z(ggml_nbytes(t), 0);
        ggml_backend_tensor_set(t, z.data(), 0, z.size());
    }
    std::printf("device %d, %d slots: gate %.1f MiB, up %.1f MiB, down %.1f MiB\n", device, S,
                ggml_nbytes(gate_w)/1048576.0, ggml_nbytes(up_w)/1048576.0, ggml_nbytes(down_w)/1048576.0);

    std::vector<req_graph> gs(9);
    for (int m = 1; m <= 8; ++m) gs[m] = build(be, gate_w, up_w, down_w, m, n_embd, n_ff);

    std::mt19937 rng(1);
    std::vector<float> x(n_embd), res(n_embd), wv(8, 0.125f);
    for (float & v : x) v = (float) (rng() % 1000) / 1000.0f - 0.5f;

    std::printf("%2s | %9s %9s %9s | %s\n", "m", "mean us", "p50 us", "p99 us", "(whole request: copy in, 4 matmul-id + swiglu + sum, copy out, synchronised)");
    for (int m = 1; m <= 8; ++m) {
        req_graph & g = gs[m];
        std::vector<double> lat;
        lat.reserve(iters);
        for (int it = -200; it < iters; ++it) {
            int32_t ids[8];
            for (int i = 0; i < m; ++i) ids[i] = (int32_t) (rng() % S);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(g.x,   x.data(), 0, n_embd*sizeof(float));
            ggml_backend_tensor_set(g.ids, ids,      0, m*sizeof(int32_t));
            ggml_backend_tensor_set(g.w,   wv.data(), 0, m*sizeof(float));
            ggml_backend_graph_compute(be, g.gf);
            ggml_backend_tensor_get(g.out, res.data(), 0, n_embd*sizeof(float));
            const auto t1 = std::chrono::steady_clock::now();
            if (it >= 0) lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(lat.begin(), lat.end());
        double mean = 0; for (double v : lat) mean += v; mean /= lat.size();
        std::printf("%2d | %9.1f %9.1f %9.1f |\n", m, mean, lat[lat.size()/2], lat[(size_t) (lat.size()*0.99)]);
    }
    return 0;
}
