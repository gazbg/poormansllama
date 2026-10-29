// Single-thread cost of the CPU expert dot products of Qwen3.6-35B-A3B (gate/up: Q4_K rows of 2048, down: Q5_K rows of 512), with the weights hot in the
// caches (one expert, repeated) and streaming from DRAM (a different expert each time), so the compute floor and the memory cost of the lean host leg can be told apart.
//   g++ -O3 -march=native -I third_party/llama.cpp/ggml/include tools/pascal/bench_vecdot.cpp -L build/llama-cuda/bin -lggml-cpu -lggml-base -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/bench_vecdot
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

static double bench(ggml_type type, int k, int rows, int n_experts, bool hot, int reps) {
    const ggml_type_traits_cpu * tt = ggml_get_type_traits_cpu(type);
    const ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(tt->vec_dot_type);
    const size_t rb = ggml_row_size(type, k);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) k*rows);
    for (auto & v : f) { v = nd(rng); }
    std::vector<char> one((size_t) rows*rb);
    ggml_quantize_chunk(type, f.data(), one.data(), 0, rows, k, nullptr);
    std::vector<char> w((size_t) n_experts*rows*rb);
    for (int e = 0; e < n_experts; ++e) {
        std::copy(one.begin(), one.end(), w.begin() + (size_t) e*rows*rb);
    }
    std::vector<float> x((size_t) k);
    for (auto & v : x) { v = nd(rng); }
    std::vector<char> xq(ggml_row_size(tt->vec_dot_type, k));
    tq->from_float(x.data(), xq.data(), k);
    float s = 0.0f, acc = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        const char * base = w.data() + (size_t) (hot ? 0 : i % n_experts)*rows*rb;
        for (int r = 0; r < rows; ++r) {
            tt->vec_dot(k, &s, 0, base + (size_t) r*rb, 0, xq.data(), 0, 1);
            acc += s;
        }
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count()/((double) reps*rows);
    if (acc == 12345.0f) { std::printf("."); }
    return ns;
}

int main() {
    ggml_cpu_init();
    const int ne = 192;   // 192 experts x ~0.6-0.7 MiB: far larger than the 8 MiB L3
    const double q4h = bench(GGML_TYPE_Q4_K, 2048,  512, ne, true,  400);
    const double q4c = bench(GGML_TYPE_Q4_K, 2048,  512, ne, false, 400);
    const double q5h = bench(GGML_TYPE_Q5_K,  512, 2048, ne, true,  400);
    const double q5c = bench(GGML_TYPE_Q5_K,  512, 2048, ne, false, 400);
    const double q6h = bench(GGML_TYPE_Q6_K,  512, 2048, ne, true,  400);
    std::printf("single thread, ns per row: q4_K k=2048 hot %.1f / streaming %.1f; q5_K k=512 hot %.1f / streaming %.1f; q6_K k=512 hot %.1f\n", q4h, q4c, q5h, q5c, q6h);
    const double exp_hot = 2*512*q4h + 2048*q5h;
    const double exp_str = 2*512*q4c + 2048*q5c;
    std::printf("one expert (gate + up + down), one thread: compute %.1f us hot, %.1f us streaming; with 4 threads: %.1f / %.1f us\n",
                exp_hot/1e3, exp_str/1e3, exp_hot/4e3, exp_str/4e3);
    return 0;
}
