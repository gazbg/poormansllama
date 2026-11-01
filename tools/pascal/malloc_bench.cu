// Plan H113: what does the allocation loop of a cache resize cost, and would fewer, larger allocations help? 40 buffers of the size of one cache layer against 20 of twice that, 10, 4 and 1;
// each variant allocates, then frees everything. The sizes are the 31-slot and 51-slot caches of the recipe (about 2.3 GB and 3.8 GB in all).
//   nvcc -O2 -std=c++17 -arch=sm_61 tools/pascal/malloc_bench.cu -o build/pascal/malloc_bench && build/pascal/malloc_bench [device]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int dev0 = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev0));
    CK(cudaFree(nullptr));
    if (std::getenv("PEER") != nullptr) {      // PEER=1 enables peer access to the other GPU, as the two-GPU prefill does (ggml enables it for every pair)
        const int other = dev0 == 0 ? 1 : 0;
        CK(cudaSetDevice(other));
        CK(cudaFree(nullptr));
        CK(cudaSetDevice(dev0));
        cudaError_t e = cudaDeviceEnablePeerAccess(other, 0);
        std::printf("peer access to device %d: %s\n", other, cudaGetErrorString(e));
        (void) cudaGetLastError();
    }
    if (const char * hr = std::getenv("HOSTREG_GIB")) {      // HOSTREG_GIB=N registers N GiB of host memory (portable), as the engine does with the expert banks
        const size_t n = (size_t) std::atoi(hr) << 30;
        void * h = std::malloc(n);
        std::memset(h, 1, n);
        const double t0 = now();
        CK(cudaHostRegister(h, n, cudaHostRegisterPortable | cudaHostRegisterReadOnly));
        std::printf("registered %.0f GiB of host memory in %.0f ms\n", n / 1073741824.0, (now() - t0) * 1e3);
    }
    const size_t layers = 40;
    // optional second argument: leave only this many MiB free beyond the largest cache (a ballast takes the rest), as the engine finds the card when it resizes
    void * ballast = nullptr;
    if (argc > 2) {
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const size_t keep = ((size_t) 3791 + (size_t) std::atoi(argv[2])) << 20;
        if (free_b > keep) {
            CK(cudaMalloc(&ballast, free_b - keep));
        }
        CK(cudaMemGetInfo(&free_b, &total_b));
        std::printf("ballast taken: %.0f MiB free now\n", free_b / 1048576.0);
    }
    for (size_t total_mib : { 2333, 3791 }) {
        const size_t total = total_mib << 20;
        std::printf("cache of %zu MiB in %zu layers (%.1f MiB each):\n", total_mib, layers, (double) total / layers / 1048576.0);
        for (size_t groups : { 40, 20, 10, 4, 1 }) {
            double best_alloc = 1e9, best_clear = 1e9, best_free = 1e9;
            for (int rep = 0; rep < 5; ++rep) {
                std::vector<void *> p(groups, nullptr);
                const size_t each = total / groups;
                const double t0 = now();
                for (size_t g = 0; g < groups; ++g) {
                    CK(cudaMalloc(&p[g], each));
                }
                const double t1 = now();
                for (size_t g = 0; g < groups; ++g) {
                    CK(cudaMemset(p[g], 0, each));       // the old clear of the whole buffer
                }
                CK(cudaDeviceSynchronize());
                const double t2 = now();
                for (size_t g = 0; g < groups; ++g) {
                    CK(cudaFree(p[g]));
                }
                const double t3 = now();
                best_alloc = std::min(best_alloc, t1 - t0);
                best_clear = std::min(best_clear, t2 - t1);
                best_free  = std::min(best_free, t3 - t2);
            }
            std::printf("  %2zu allocation(s): malloc %6.1f ms, memset of everything %6.1f ms, free %6.1f ms\n", groups, best_alloc * 1e3, best_clear * 1e3, best_free * 1e3);
        }
    }
    // the pattern of the engine's resize: an old cache of 40 layers exists; for every layer a new buffer is allocated, filled by a device copy out of the old one, and the old one is freed.
    // `lag` = how many layers behind the frees run (0: at once, as the engine does; 4, 8: deferred in a batch)
    std::printf("resize pattern: 40 old layers of 94.8 MiB -> 40 new layers of 58.3 MiB (a shrink), time spent in cudaMalloc / cudaFree:\n");
    for (int lag : { 0, 4, 8, 40 }) {
        double best_m = 1e9, best_f = 1e9, best_all = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            const size_t old_sz = (size_t) (94.8 * 1048576), new_sz = (size_t) (58.3 * 1048576);
            std::vector<void *> o(layers), n(layers, nullptr);
            for (size_t i = 0; i < layers; ++i) { CK(cudaMalloc(&o[i], old_sz)); CK(cudaMemset(o[i], 1, old_sz)); }
            CK(cudaDeviceSynchronize());
            cudaStream_t st;
            CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            double tm = 0, tf = 0;
            const double t0 = now();
            for (size_t i = 0; i < layers; ++i) {
                double a = now();
                CK(cudaMalloc(&n[i], new_sz));
                tm += now() - a;
                CK(cudaMemcpyAsync(n[i], o[i], new_sz, cudaMemcpyDeviceToDevice, st));
                if ((int) i >= lag || lag == 0) {
                    const size_t j = lag == 0 ? i : i - lag;
                    CK(cudaStreamSynchronize(st));
                    a = now();
                    CK(cudaFree(o[j]));
                    tf += now() - a;
                    o[j] = nullptr;
                }
            }
            CK(cudaStreamSynchronize(st));
            for (size_t i = 0; i < layers; ++i) {
                if (o[i] != nullptr) { double a = now(); CK(cudaFree(o[i])); tf += now() - a; }
            }
            const double total = now() - t0;
            for (size_t i = 0; i < layers; ++i) CK(cudaFree(n[i]));
            CK(cudaStreamDestroy(st));
            best_m = std::min(best_m, tm); best_f = std::min(best_f, tf); best_all = std::min(best_all, total);
        }
        std::printf("  frees %2d layers behind: cudaMalloc %6.1f ms, cudaFree %6.1f ms, whole loop %6.1f ms\n", lag, best_m * 1e3, best_f * 1e3, best_all * 1e3);
    }
    return 0;
}
