// Host DRAM read bandwidth with N threads (AVX2 loads, one private 512 MiB buffer per thread), for a fixed time.
//   nvcc -O2 -arch=sm_120 -Xcompiler "/arch:AVX2" -o build/dense/dram-bw.exe tools/dense/dram-bw.cu
//   build/dense/dram-bw.exe THREADS SECONDS
// Run next to tools/dense/h2d-loop.cu to see how the copy engine's host reads and the CPU's reads share the DRAM.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 16;
    double seconds = argc > 2 ? atof(argv[2]) : 5.0;
    const size_t bytes = (size_t) 512 << 20;
    std::atomic<bool> go{false}, stop{false};
    std::vector<double> sums(n, 0.0);
    std::vector<std::thread> th;
    std::vector<unsigned long long> done(n, 0);
    for (int t = 0; t < n; ++t) {
        th.emplace_back([&, t] {
            char * buf = (char *) _aligned_malloc(bytes, 64);
            for (size_t i = 0; i < bytes; i += 4096) buf[i] = (char) i;
            while (!go.load()) std::this_thread::yield();
            __m256i acc = _mm256_setzero_si256();
            unsigned long long total = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < bytes; i += 128) {
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i)));
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 32)));
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 64)));
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 96)));
                }
                total += bytes;
                if (stop.load(std::memory_order_relaxed)) break;
            }
            sums[t] = (double) _mm256_extract_epi64(acc, 0);
            done[t] = total;
        });
    }
    auto t0 = std::chrono::steady_clock::now();
    go = true;
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto & x : th) x.join();
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long tot = 0;
    for (auto d : done) tot += d;
    printf("dram-bw: %d threads, %.2f GB/s (whole passes only, %.1f s)\n", n, tot / 1e9 / dt, dt);
    return 0;
}
