// The CPU expert leg of Qwen3.6-35B-A3B as a microbenchmark: a team of T threads, each reading its row slice of m random experts (gate and up: Q4_K rows of 2048, then down: Q5_K rows of 512) with the real
// vec_dot kernels, a spin barrier between the two stages, over a 2 GiB region of "experts" backed by 4 KiB pages or by 2 MiB pages (madvise MADV_HUGEPAGE). Prints microseconds per request by m.
//   g++ -O3 -march=native -std=c++17 -pthread -I third_party/llama.cpp/ggml/include tools/pascal/bench_stream.cpp -L build/llama-cuda/bin -lggml-cpu -lggml-base -Wl,-rpath,$PWD/build/llama-cuda/bin -o build/pascal/bench_stream
//   build/pascal/bench_stream [threads=4] [requests=3000]
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <x86intrin.h>

struct spin_barrier {
    std::atomic<int> count{0};
    std::atomic<int> sense{0};
    int n = 1;
    void wait(int & local) {
        local ^= 1;
        if (count.fetch_add(1) == n - 1) {
            count.store(0);
            sense.store(local);
        } else {
            while (sense.load() != local) { _mm_pause(); }
        }
    }
};

static const int N_EXP = 1024;   // experts in the region
static const int K_UD = 2048, ROWS_UD = 512, K_D = 512, ROWS_D = 2048;

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4;
    const int REQS = argc > 2 ? atoi(argv[2]) : 3000;
    ggml_cpu_init();
    const ggml_type_traits_cpu * tq4 = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const ggml_type_traits_cpu * tq5 = ggml_get_type_traits_cpu(GGML_TYPE_Q5_K);
    const ggml_type_traits_cpu * tx  = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t rb4 = ggml_row_size(GGML_TYPE_Q4_K, K_UD), rb5 = ggml_row_size(GGML_TYPE_Q5_K, K_D);
    const size_t bank_ud = rb4*ROWS_UD, bank_d = rb5*ROWS_D, expert_bytes = 2*bank_ud + bank_d;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f((size_t) K_UD*ROWS_UD);
    for (auto & v : f) { v = nd(rng); }
    std::vector<char> ud(bank_ud), dn(bank_d);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), ud.data(), 0, ROWS_UD, K_UD, nullptr);
    std::vector<float> fd((size_t) K_D*ROWS_D);
    for (auto & v : fd) { v = nd(rng); }
    ggml_quantize_chunk(GGML_TYPE_Q5_K, fd.data(), dn.data(), 0, ROWS_D, K_D, nullptr);
    std::vector<float> x_ud(K_UD), x_d(K_D);
    for (auto & v : x_ud) { v = nd(rng); }
    for (auto & v : x_d) { v = nd(rng); }
    std::vector<char> xq_ud(ggml_row_size(GGML_TYPE_Q8_K, K_UD)), xq_d(ggml_row_size(GGML_TYPE_Q8_K, K_D));
    tx->from_float(x_ud.data(), xq_ud.data(), K_UD);
    tx->from_float(x_d.data(), xq_d.data(), K_D);

    for (int mode = 0; mode < 2; ++mode) {
        const size_t total = (size_t) N_EXP*expert_bytes;
        char * base = nullptr;
        if (mode == 0) {
            base = (char *) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            madvise(base, total, MADV_NOHUGEPAGE);
        } else {
            void * p = nullptr;
            posix_memalign(&p, 2u << 20, total);
            base = (char *) p;
            madvise(base, total, MADV_HUGEPAGE);
        }
        for (int e = 0; e < N_EXP; ++e) {
            char * eb = base + (size_t) e*expert_bytes;
            memcpy(eb, ud.data(), bank_ud);
            memcpy(eb + bank_ud, ud.data(), bank_ud);
            memcpy(eb + 2*bank_ud, dn.data(), bank_d);
        }
        long huge_kb = 0;
        if (FILE * fp = fopen("/proc/self/smaps_rollup", "r")) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) { if (sscanf(line, "AnonHugePages: %ld kB", &huge_kb) == 1) { break; } }
            fclose(fp);
        }
        std::printf("%s pages (AnonHugePages %ld MiB), %d threads: us per request by experts per request m\n", mode ? "2 MiB" : "4 KiB", huge_kb/1024, T);

        for (int m : { 1, 2, 4, 6, 8 }) {
            spin_barrier bar; bar.n = T;
            std::atomic<int> go{0}, done{0};
            std::vector<int> plan((size_t) REQS*m);
            for (auto & v : plan) { v = (int) (rng() % N_EXP); }
            std::vector<double> t_req;
            auto worker = [&](int ith) {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(ith, &set);
                sched_setaffinity(0, sizeof(set), &set);
                int ls = 0;
                float s = 0.0f, acc = 0.0f;
                for (int r = 0; r < REQS; ++r) {
                    bar.wait(ls);
                    const auto t0 = std::chrono::steady_clock::now();
                    const int r0 = ith*ROWS_UD/T, r1 = (ith + 1)*ROWS_UD/T, d0 = ith*ROWS_D/T, d1 = (ith + 1)*ROWS_D/T;
                    for (int i = 0; i < m; ++i) {
                        const char * eb = base + (size_t) plan[(size_t) r*m + i]*expert_bytes;
                        for (int row = r0; row < r1; ++row) {
                            tq4->vec_dot(K_UD, &s, 0, eb + (size_t) row*rb4, 0, xq_ud.data(), 0, 1); acc += s;
                            tq4->vec_dot(K_UD, &s, 0, eb + bank_ud + (size_t) row*rb4, 0, xq_ud.data(), 0, 1); acc += s;
                        }
                    }
                    bar.wait(ls);
                    for (int i = 0; i < m; ++i) {
                        const char * eb = base + (size_t) plan[(size_t) r*m + i]*expert_bytes + 2*bank_ud;
                        for (int row = d0; row < d1; ++row) {
                            tq5->vec_dot(K_D, &s, 0, eb + (size_t) row*rb5, 0, xq_d.data(), 0, 1); acc += s;
                        }
                    }
                    bar.wait(ls);
                    if (ith == 0) {
                        t_req.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                    }
                }
                if (acc == 1234.5f) { std::printf("."); }
            };
            std::vector<std::thread> th;
            for (int i = 1; i < T; ++i) { th.emplace_back(worker, i); }
            worker(0);
            for (auto & t : th) { t.join(); }
            double sum = 0.0;   // the first tenth of the requests warms the caches and is left out of the mean
            for (size_t i = REQS/10; i < t_req.size(); ++i) { sum += t_req[i]; }
            const double mean = sum/(double) (t_req.size() - REQS/10);
            std::sort(t_req.begin(), t_req.end());
            std::printf("  m=%d: mean %.1f us, median %.1f us\n", m, mean, t_req[t_req.size()/2]);
        }
        if (mode == 0) { munmap(base, total); } else { free(base); }
    }
    return 0;
}
