// Causal-window flash attention at the Qwen3.6 prefill shape (head size 256, 16 query heads, 2 KV heads, q8_0 K and V), fixed random data, output written to a file.
// Run it twice (e.g. with and without LLAMA_FATTN_TILE_Q8PV=1) and compare the files with the second mode `cmp A B`: prints the error by query head and by query.
//   g++ -O2 -I third_party/llama.cpp/ggml/include tools/pascal/fa_pv8_check.cpp -L build/llama-cuda/bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
//       -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/fa_pv8_check
//   LLAMA_FATTN_TILE_Q8QK=1 build/pascal/fa_pv8_check run out_a.bin [kv] [nq]
//   LLAMA_FATTN_TILE_Q8QK=1 LLAMA_FATTN_TILE_Q8PV=1 build/pascal/fa_pv8_check run out_b.bin [kv] [nq]
//   build/pascal/fa_pv8_check cmp out_a.bin out_b.bin [kv] [nq]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3 || (std::string(argv[1]) == "cmp" && argc < 4)) { std::printf("usage: run|cmp file [file2] [kv] [nq]\n"); return 2; }
    const bool cmp = std::string(argv[1]) == "cmp";
    const int  ia  = cmp ? 4 : 3;
    const int  kv  = argc > ia     ? atoi(argv[ia])     : 512;
    const int  nq  = argc > ia + 1 ? atoi(argv[ia + 1]) : 128;
    const int  hs = 256, nkv = 2, nh = 16;
    const size_t n_out = (size_t) hs*nh*nq;
    if (cmp) {
        std::vector<float> a(n_out), b(n_out);
        FILE * fa = fopen(argv[2], "rb"); FILE * fb = fopen(argv[3], "rb");
        if (!fa || !fb || fread(a.data(), 4, n_out, fa) != n_out || fread(b.data(), 4, n_out, fb) != n_out) { std::printf("read error\n"); return 2; }
        double num = 0, den = 0;
        for (size_t i = 0; i < n_out; ++i) { num += (double) (a[i] - b[i])*(a[i] - b[i]); den += (double) b[i]*b[i]; }
        std::printf("NMSE %.3e\n", num/den);
        std::printf("per query (sum over heads), worst 8:\n");
        std::vector<std::pair<double, int>> w;
        for (int q = 0; q < nq; ++q) {
            double e = 0, d = 0;
            for (int h = 0; h < nh; ++h) { for (int i = 0; i < hs; ++i) { const size_t k = ((size_t) q*nh + h)*hs + i; e += (double) (a[k] - b[k])*(a[k] - b[k]); d += (double) b[k]*b[k]; } }
            w.push_back({ e/d, q });
        }
        std::sort(w.begin(), w.end());
        for (int i = 0; i < 8; ++i) { std::printf("  query %3d: nmse %.3e\n", w[w.size() - 1 - i].second, w[w.size() - 1 - i].first); }
        std::printf("per head:");
        for (int h = 0; h < nh; ++h) {
            double e = 0, d = 0;
            for (int q = 0; q < nq; ++q) { for (int i = 0; i < hs; ++i) { const size_t k = ((size_t) q*nh + h)*hs + i; e += (double) (a[k] - b[k])*(a[k] - b[k]); d += (double) b[k]*b[k]; } }
            std::printf(" %.1e", e/d);
        }
        std::printf("\nper head dim block (8 blocks of 32):");
        for (int blk = 0; blk < 8; ++blk) {
            double e = 0, d = 0;
            for (int q = 0; q < nq; ++q) { for (int h = 0; h < nh; ++h) { for (int i = 0; i < 32; ++i) { const size_t k = ((size_t) q*nh + h)*hs + blk*32 + i; e += (double) (a[k] - b[k])*(a[k] - b[k]); d += (double) b[k]*b[k]; } } }
            std::printf(" %.1e", e/d);
        }
        std::printf("\n");
        return 0;
    }

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::printf("no CUDA backend\n"); return 2; }
    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, nq, nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, hs, kv, nkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, hs, kv, nkv, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f/sqrtf((float) hs), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_causal(out, kv - nq, kv);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> qf((size_t) hs*nq*nh), kf((size_t) hs*kv*nkv), vf((size_t) hs*kv*nkv);
    for (auto & x : qf) { x = ud(rng); }
    for (auto & x : kf) { x = ud(rng); }
    for (auto & x : vf) { x = ud(rng); }
    std::vector<char> kq(ggml_nbytes(k)), vq(ggml_nbytes(v));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, kf.data(), kq.data(), 0, (int64_t) kv*nkv, hs, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, vf.data(), vq.data(), 0, (int64_t) kv*nkv, hs, nullptr);
    ggml_backend_tensor_set(q, qf.data(), 0, qf.size()*4);
    ggml_backend_tensor_set(k, kq.data(), 0, kq.size());
    ggml_backend_tensor_set(v, vq.data(), 0, vq.size());
    ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    std::vector<float> res(n_out);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*4);
    FILE * f = fopen(argv[2], "wb"); fwrite(res.data(), 4, res.size(), f); fclose(f);
    std::printf("wrote %zu floats, first %.5f %.5f\n", res.size(), res[0], res[1]);
    return 0;
}
