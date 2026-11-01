// H120 step 1: device <-> host copies of a sequence-state sized blob (default 1.1 GiB) through three kinds of host memory:
//   A a freshly allocated, zero-filled std::vector (what the pass does today), B malloc'd pageable memory, C page-locked memory (cudaHostRegister on a malloc'd range, as the pass would keep it)
// build: nvcc -O2 -o build/pascal/d2h_bench tools/pascal/d2h_bench.cu ; run: build/pascal/d2h_bench [MiB] [device]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char ** argv) {
    const size_t mib   = argc > 1 ? (size_t) std::atoll(argv[1]) : 1126;
    const int    dev   = argc > 2 ? std::atoi(argv[2]) : 0;
    const size_t bytes = mib << 20;
    const size_t chunk = 64u << 20;       // the state is written tensor by tensor: a few dozen copies
    CK(cudaSetDevice(dev));
    void * d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemset(d, 0x5a, bytes));
    CK(cudaDeviceSynchronize());

    auto d2h = [&](uint8_t * dst) { for (size_t o = 0; o < bytes; o += chunk) { CK(cudaMemcpy(dst + o, (uint8_t *) d + o, std::min(chunk, bytes - o), cudaMemcpyDeviceToHost)); } return 0; };
    auto h2d = [&](const uint8_t * src) { for (size_t o = 0; o < bytes; o += chunk) { CK(cudaMemcpy((uint8_t *) d + o, src + o, std::min(chunk, bytes - o), cudaMemcpyHostToDevice)); } return 0; };

    for (int rep = 0; rep < 3; ++rep) {
        // A: a new std::vector every time, zero-filled by resize()
        double t0 = now_ms();
        std::vector<uint8_t> v;
        v.resize(bytes);
        double t1 = now_ms();
        if (d2h(v.data())) return 1;
        double t2 = now_ms();
        if (h2d(v.data())) return 1;
        double t3 = now_ms();
        std::printf("rep %d A vector     : alloc+zero %6.0f ms, d2h %6.0f ms (%.2f GB/s), h2d %6.0f ms (%.2f GB/s), total %6.0f ms\n", rep, t1 - t0, t2 - t1, bytes / 1e6 / (t2 - t1), t3 - t2, bytes / 1e6 / (t3 - t2), t3 - t0);
        v = std::vector<uint8_t>();

        // B: malloc'd pageable, not touched before
        t0 = now_ms();
        uint8_t * b = (uint8_t *) std::malloc(bytes);
        t1 = now_ms();
        if (d2h(b)) return 1;
        t2 = now_ms();
        if (h2d(b)) return 1;
        t3 = now_ms();
        std::printf("rep %d B malloc     : alloc      %6.0f ms, d2h %6.0f ms (%.2f GB/s), h2d %6.0f ms (%.2f GB/s), total %6.0f ms\n", rep, t1 - t0, t2 - t1, bytes / 1e6 / (t2 - t1), t3 - t2, bytes / 1e6 / (t3 - t2), t3 - t0);
        std::free(b);
    }

    // C: page-locked, registered once and reused
    uint8_t * c = nullptr;
    if (posix_memalign((void **) &c, 4096, bytes) != 0) { return 1; }
    double t0 = now_ms();
    CK(cudaHostRegister(c, bytes, cudaHostRegisterDefault));
    double t1 = now_ms();
    std::printf("C registration (once): %6.0f ms for %zu MiB\n", t1 - t0, mib);
    for (int rep = 0; rep < 3; ++rep) {
        double a = now_ms();
        if (d2h(c)) return 1;
        double b2 = now_ms();
        if (h2d(c)) return 1;
        double e = now_ms();
        std::printf("rep %d C pinned     :                  d2h %6.0f ms (%.2f GB/s), h2d %6.0f ms (%.2f GB/s), total %6.0f ms\n", rep, b2 - a, bytes / 1e6 / (b2 - a), e - b2, bytes / 1e6 / (e - b2), e - a);
    }
    cudaHostUnregister(c);
    std::free(c);
    cudaFree(d);
    return 0;
}
