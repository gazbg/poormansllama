// Relocated from build/gplr29-pinprobe.cu (2026-09-16 provenance sweep; the T-GPLR-34
// convention: committed runners/probes live under tools/, with paths made
// relative to the repository root so a clean checkout reproduces the run).
// Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
// T-GPLR-29: pinned-pool probe (throwaway microbenchmark, outside the vendored
// tree). Measures on this box (RTX 5090, WDDM, PCIe Gen5 x16, CUDA 13.2):
//   A  pinned (cudaHostAlloc) -> VRAM DMA ceiling
//   B  direct: mapped-GGUF (pageable) -> VRAM, warm  [control: must land near
//      the recorded ~4.4 GB/s warm per-copy rate, else the probe is invalid]
//      plus a one-shot cold/first-touch figure
//   C  the T-GPLR-18 two-hop chain: memcpy(map->pinned) + pinned DMA
//   D  whether cudaHostRegister accepts the mapped view under WDDM
// One bank-sized (default 453 MB) region mapped from the real MoE GGUF so the
// pageable case IS our production copy source.
//
// Usage: gplr29-pinprobe.exe <model.gguf> [offset_bytes=1GiB] [bytes=453e6] [iters=5]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>
#include <windows.h>
#include <cuda_runtime.h>

static void check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// one timed cudaMemcpy + device sync; returns elapsed ms
static double timed_copy(void * dst, const void * src, size_t bytes) {
    const double t0 = now_ms();
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "bench copy");
    check(cudaDeviceSynchronize(), "sync");
    return now_ms() - t0;
}

static void bench(const char * label, void * dst, const void * src, size_t bytes, int iters) {
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
        ms.push_back(timed_copy(dst, src, bytes));
    }
    const double best  = *std::min_element(ms.begin(), ms.end());
    const double med   = median_of(ms);
    printf("%-38s bytes=%zu  best=%.1f ms (%7.2f GB/s)  median=%.1f ms (%7.2f GB/s)\n",
           label, bytes, best, bytes / (best * 1e6), med, bytes / (med * 1e6));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [offset_bytes=1073741824] [bytes=453000000] [iters=5]\n", argv[0]);
        return 2;
    }
    const size_t offset = argc > 2 ? (size_t) atoll(argv[2]) : (size_t) 1073741824;
    size_t       bytes  = argc > 3 ? (size_t) atoll(argv[3]) : (size_t) 453000000;
    const int    iters  = argc > 4 ? atoi(argv[4]) : 5;
    bytes &= ~(size_t) 4095; // page-align for cleanliness

    int dev = 0;
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, dev), "props");
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("device=%s sm_%d%d  driver=%d.%d runtime=%d.%d  (WDDM consumer part)\n",
           prop.name, prop.major, prop.minor, drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);

    // --- map the real model file (the production copy source) ---
    HANDLE h = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "CreateFile failed err=%lu\n", GetLastError()); return 1; }
    HANDLE fm = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!fm) { fprintf(stderr, "CreateFileMapping failed err=%lu\n", GetLastError()); return 1; }
    BYTE * map = (BYTE *) MapViewOfFile(fm, FILE_MAP_READ, 0, 0, 0);
    if (!map) { fprintf(stderr, "MapViewOfFile failed err=%lu\n", GetLastError()); return 1; }
    BYTE * region = map + offset;
    printf("mapped %s, probing region at offset %zu (%zu bytes)\n", argv[1], offset, bytes);

    void * dptr = nullptr;
    check(cudaMalloc(&dptr, bytes), "cudaMalloc");

    // --- B0: cold / first-touch one-shot from the mapped region ---
    {
        const double ms = timed_copy(dptr, region, bytes);
        printf("%-38s bytes=%zu  first-touch=%.1f ms (%7.2f GB/s)\n",
               "B0 direct mmap->VRAM COLD", bytes, ms, bytes / (ms * 1e6));
    }

    // --- B: warm direct (pageable) ---
    bench("B  direct mmap->VRAM WARM", dptr, region, bytes, iters);

    // --- A: pinned pool DMA ceiling ---
    void * pinned = nullptr;
    cudaError_t pin_err = cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault);
    if (pin_err != cudaSuccess) {
        printf("%-38s FAILED: %s\n", "A  pinned pool alloc", cudaGetErrorString(pin_err));
    } else {
        bench("A  pinned cudaHostAlloc->VRAM", dptr, pinned, bytes, iters);

        // --- C: the T-GPLR-18 two-hop chain ---
        // first hop: warm CPU memcpy map->pinned (pages already in WS from B)
        {
            std::vector<double> ms;
            memcpy(pinned, region, bytes); // warm the chain
            for (int i = 0; i < iters; ++i) {
                const double t0 = now_ms();
                memcpy(pinned, region, bytes);
                ms.push_back(now_ms() - t0);
            }
            const double best = *std::min_element(ms.begin(), ms.end());
            const double med  = median_of(ms);
            printf("%-38s bytes=%zu  best=%.1f ms (%7.2f GB/s)  median=%.1f ms (%7.2f GB/s)\n",
                   "C1 memcpy mmap->pinned (CPU)", bytes, best, bytes / (best * 1e6), med, bytes / (med * 1e6));
            // chain = C1 memcpy + A DMA (components measured independently)
            double dma_best = 0;
            {
                std::vector<double> dms;
                for (int i = 0; i < iters; ++i) dms.push_back(timed_copy(dptr, pinned, bytes));
                dma_best = *std::min_element(dms.begin(), dms.end());
            }
            const double chain = med + dma_best;
            printf("%-38s bytes=%zu  chain(median+best)=%.1f ms (%7.2f GB/s)  [vs direct-warm above]\n",
                   "C  T-GPLR-18 chain effective", bytes, chain, bytes / (chain * 1e6));
        }
        check(cudaFreeHost(pinned), "cudaFreeHost");
    }

    // --- D: can WDDM CUDA register the mapped view? (do last; may invalidate) ---
    {
        cudaError_t e = cudaHostRegister(region, bytes, cudaHostRegisterDefault);
        if (e == cudaSuccess) {
            printf("%-38s SUCCESS (unexpected on file-backed views; benchmarking)\n", "D  cudaHostRegister on mmap");
            bench("D1 registered mmap->VRAM", dptr, region, bytes, iters);
            check(cudaHostUnregister(region), "unregister");
        } else {
            printf("%-38s FAILED: %s (matches the Linux-documented behavior)\n", "D  cudaHostRegister on mmap", cudaGetErrorString(e));
        }
    }

    check(cudaFree(dptr), "cudaFree");
    UnmapViewOfFile(map);
    CloseHandle(fm);
    CloseHandle(h);
    printf("probe done\n");
    return 0;
}
