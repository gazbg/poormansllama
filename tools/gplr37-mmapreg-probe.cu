// T-GPLR-37 probe (stage 1, feasibility): can a bank-sized window of a
// FILE-BACKED mmap be page-locked with cudaHostRegister, and does a DMA sourced
// from those registered pages run at the pinned ceiling without the CPU hop?
//
// Why this exists: the layer-residency staging path (T-GPLR-33 stage_members)
// always hops each member into a pinned staging buffer with a CPU memcpy, then
// DMAs from the pinned copy. The recon measured that hop at ~39 ms/bank
// (~1.6 s per 4k pass, 40 banks); `--copy-source direct` skips the hop but
// DMAs from pageable mmap memory at ~4.4 GB/s instead of the 14.4 GB/s pinned
// ceiling. This probe measures the third option the path does not have:
// register the mmap'd source region, then DMA straight from it.
//
// Arms (each 3 reps, steady-state median reported):
//   A pageable : cudaMemcpyAsync from the raw mmap window (== --copy-source direct)
//   B registered: cudaHostRegister(window) then the same DMA (registration timed
//                 separately; unregistered at the end)
//   C pinned-hop : memcpy(mmap -> cudaHostAlloc buffer) timed, then DMA from it
//                 (== the current --copy-source pinned path, hop included)
// Correctness: the first 1 MiB and last 1 MiB of each arm's device buffer are
// copied back and compared byte-for-byte against the file window, so a "fast"
// arm that staged wrong bytes cannot be reported as a win.
//
// Build: nvcc -O2 -arch=native -o build/gplr37-mmapreg-probe.exe tools/gplr37-mmapreg-probe.cu
// Run  : tools/run-gplr37-probe.ps1 (under the GPU mutex; ~10 s of GPU work)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    return 2; } } while (0)

static double now_ms() {
    using clock_t_ = std::chrono::steady_clock;
    static const clock_t_::time_point t0 = clock_t_::now();
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
}

struct mapped {
    void * base = nullptr;
    size_t bytes = 0;
    bool ok = false;
};

static mapped map_file(const char * path, size_t offset, size_t bytes) {
    mapped m;
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", path); return m; }
    HANDLE hm = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (hm == nullptr) { fprintf(stderr, "CreateFileMapping failed\n"); CloseHandle(h); return m; }
    void * p = MapViewOfFile(hm, FILE_MAP_READ, (DWORD) (offset >> 32), (DWORD) (offset & 0xffffffffu), bytes);
    CloseHandle(hm); CloseHandle(h);   // the view stays valid
    if (p == nullptr) { fprintf(stderr, "MapViewOfFile failed (offset %zu)\n", offset); return m; }
    m.base = p; m.bytes = bytes; m.ok = true;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); return m; }
    void * p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, (off_t) offset);
    close(fd);
    if (p == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return m; }
    m.base = p; m.bytes = bytes; m.ok = true;
#endif
    return m;
}

static double dma_ms(void * dst, const void * src, size_t bytes, int reps, bool * ok) {
    std::vector<double> t;
    *ok = true;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaStreamSynchronize(nullptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "  dma failed: %s\n", cudaGetErrorString(err));
            *ok = false; return -1.0;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        t.push_back((double) ms);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static bool verify(void * dev, const unsigned char * file_win, size_t bytes) {
    std::vector<unsigned char> back(std::min<size_t>(bytes, 1u << 20));
    if (cudaMemcpy(back.data(), dev, back.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (memcmp(back.data(), file_win, back.size()) != 0) return false;
    const size_t tail = bytes - back.size();
    if (cudaMemcpy(back.data(), (const char *) dev + tail, back.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return memcmp(back.data(), file_win + tail, back.size()) == 0;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf";
    size_t bytes  = argc > 2 ? (size_t) strtoull(argv[2], nullptr, 0) : (size_t) 453u * 1024u * 1024u;
    size_t offset = argc > 3 ? (size_t) strtoull(argv[3], nullptr, 0) : (size_t) 1024u * 1024u * 1024u;
    const size_t page = 4096;
    offset &= ~(page - 1);
    bytes   = (bytes + page - 1) & ~(page - 1);

    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    size_t freeb = 0, totalb = 0;
    CK(cudaMemGetInfo(&freeb, &totalb));
    printf("device: %s | free %.2f GiB / total %.2f GiB | asyncEngineCount %d\n",
           prop.name, freeb / 1073741824.0, totalb / 1073741824.0, prop.asyncEngineCount);
    printf("window: %s offset %zu bytes %zu (%.1f MiB)\n\n", path, offset, bytes, bytes / 1048576.0);

    mapped m = map_file(path, offset, bytes);
    if (!m.ok) return 2;
    const unsigned char * src = (const unsigned char *) m.base;

    void * dst = nullptr;
    CK(cudaMalloc(&dst, bytes));

    bool ok = false;
    double dA = dma_ms(dst, src, bytes, 3, &ok);
    const bool vA = ok && verify(dst, src, bytes);
    printf("A pageable mmap DMA       : %8.2f ms  %6.2f GB/s  verify %s\n",
           dA, dA > 0 ? (bytes / 1e9) / (dA / 1e3) : 0.0, vA ? "OK" : "MISMATCH");

    // B: register the file-backed mapping, then repeat the same DMA
    const double t_reg0 = now_ms();
    cudaError_t reg_err = cudaHostRegister((void *) src, bytes,
                                           cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    const double reg_ms = now_ms() - t_reg0;
    double dB = -1.0; bool vB = false; bool registered = (reg_err == cudaSuccess);
    if (registered) {
        dB = dma_ms(dst, src, bytes, 3, &ok);
        vB = ok && verify(dst, src, bytes);
        printf("B registered mmap DMA     : %8.2f ms  %6.2f GB/s  verify %s  (registration %.2f ms)\n",
               dB, dB > 0 ? (bytes / 1e9) / (dB / 1e3) : 0.0, vB ? "OK" : "MISMATCH", reg_ms);
    } else {
        printf("B registered mmap DMA     : FAILED to register -> %s (%d)\n",
               cudaGetErrorString(reg_err), (int) reg_err);
    }

    // C: the current path -- CPU memcpy into a pinned buffer, then DMA from it
    void * pinned = nullptr;
    cudaError_t perr = cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault);
    double hopMs = -1.0, dC = -1.0; bool vC = false;
    if (perr == cudaSuccess) {
        std::vector<double> hops;
        for (int i = 0; i < 3; ++i) {
            const double t0 = now_ms();
            memcpy(pinned, src, bytes);
            hops.push_back(now_ms() - t0);
        }
        std::sort(hops.begin(), hops.end());
        hopMs = hops[1];
        dC = dma_ms(dst, pinned, bytes, 3, &ok);
        vC = ok && verify(dst, src, bytes);
        printf("C pinned hop %8.2f ms + DMA %7.2f ms (%.2f GB/s) verify %s\n",
               hopMs, dC, dC > 0 ? (bytes / 1e9) / (dC / 1e3) : 0.0, vC ? "OK" : "MISMATCH");
    } else {
        printf("C pinned hop              : cudaHostAlloc failed -> %s\n", cudaGetErrorString(perr));
    }

    // Registration cost, measured on a second window (so the first window's
    // pages are already resident and cannot inflate the number).
    if (registered) {
        const size_t off2 = offset + bytes;
        mapped m2 = map_file(path, off2, bytes);
        if (m2.ok) {
            const double t0 = now_ms();
            cudaError_t e2 = cudaHostRegister(m2.base, bytes, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
            const double t_reg = now_ms() - t0;
            printf("\nregistration of a second %.0f MiB window: %.2f ms (%s)\n",
                   bytes / 1048576.0, t_reg, e2 == cudaSuccess ? "OK" : cudaGetErrorString(e2));
            if (e2 == cudaSuccess) cudaHostUnregister(m2.base);
        }
        cudaHostUnregister((void *) src);
        printf("(unregistered; second window worth of timing above)\n");
    }

    printf("\nVERDICT INPUTS: hop cost %.2f ms/bank vs pinned DMA %.2f ms vs pageable DMA %.2f ms "
           "vs registered DMA %.2f ms\n", hopMs, dC, dA, dB);
    if (registered && dB > 0) {
        printf("registered arm removes the hop: %.2f ms saved per bank = %.1f ms per 40-bank pass\n",
               hopMs, hopMs * 40);
    }
    cudaFree(dst);
    if (pinned) cudaFreeHost(pinned);
    return 0;
}
