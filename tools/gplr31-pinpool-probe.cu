// Relocated from build/gplr31-pinpool-probe.cu (2026-09-16 provenance sweep; the T-GPLR-34
// convention: committed runners/probes live under tools/, with paths made
// relative to the repository root so a clean checkout reproduces the run).
// Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
// T-GPLR-31: does the driver grant a full-pool-size pinned allocation here?
// Throwaway probe (outside the vendored tree). Four cases:
//   A) incremental ladder: 432 MiB cudaHostAlloc blocks, one at a time, until
//      failure -> the true pool ceiling + exact cudaError
//   B) single-shot ladder: one big cudaHostAlloc of 1/4/8/12/16/17/17.5/18/20 GiB
//      -> per-allocation cap or fragmentation vs the incremental total
//   C) cudaHostRegister on plain malloc memory (anonymous pages) -> separates
//      T-GPLR-29's file-backed-mmap OOM from registration generally
//   D) GlobalMemoryStatusEx before/after -> OS commit vs driver quota
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <windows.h>
#include <cuda_runtime.h>

static const char* errname(cudaError_t e) {
    const char* s = cudaGetErrorName(e);
    return s ? s : "UNKNOWN";
}
static void print_mem_status(const char* tag) {
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    printf("[mem %s] physLoad=%lu%% availPhys=%.2f GiB totalPhys=%.2f GiB availPage=%.2f GiB\n",
           tag, (unsigned long) ms.dwMemoryLoad,
           ms.ullAvailPhys / 1073741824.0, ms.ullTotalPhys / 1073741824.0,
           ms.ullAvailPageFile / 1073741824.0);
}
static void touch_all(char* p, size_t bytes) {
    // commit the pages so "allocated" means "RAM actually pinned"
    volatile char sink = 0;
    for (size_t i = 0; i < bytes; i += 4096) sink ^= p[i];
}
static void free_all(std::vector<void*>& blocks, std::vector<size_t>& sizes) {
    for (size_t i = 0; i < blocks.size(); ++i) cudaFreeHost(blocks[i]);
    blocks.clear(); sizes.clear();
}

int main() {
    int dev = 0; cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) { printf("cudaSetDevice failed: %s\n", errname(e)); return 1; }
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("device: %s\n", prop.name);

    const size_t BANK = (size_t) 432 * 1024 * 1024;   // 432 MiB = one Qwen3.6 bank
    std::vector<void*> blocks; std::vector<size_t> sizes;

    // --- case D0: OS status before ---
    print_mem_status("before");

    // --- case A: incremental 432 MiB ladder to failure ---
    printf("\n== A) incremental cudaHostAlloc ladder (432 MiB steps) ==\n");
    size_t total = 0; int n_ok = 0; cudaError_t last = cudaSuccess;
    for (int i = 1; i <= 160; ++i) {
        void* p = nullptr;
        cudaError_t ae = cudaHostAlloc(&p, BANK, cudaHostAllocDefault);
        if (ae != cudaSuccess) { last = ae; printf("  step %2d (%6.2f GiB cumulative): FAIL %s\n",
            i, total / 1073741824.0, errname(ae));
            const char* str = cudaGetErrorString(ae);
            printf("    detail: %s\n", str ? str : "(none)");
            break; }
        touch_all((char*) p, BANK);
        blocks.push_back(p); sizes.push_back(BANK);
        total += BANK; n_ok = i;
        if (i % 10 == 0) printf("  step %2d: OK, cumulative %.2f GiB\n", i, total / 1073741824.0);
    }
    printf("  A RESULT: %d blocks OK (%.2f GiB pinned), first failure: %s\n",
           n_ok, total / 1073741824.0, errname(last));
    print_mem_status("after-ladder");
    free_all(blocks, sizes);
    print_mem_status("after-ladder-free");

    // --- case B: single-shot big allocations (fresh each, freed after) ---
    printf("\n== B) single-shot cudaHostAlloc ladder ==\n");
    const double gib[] = { 1, 4, 8, 12, 16, 17, 17.5, 18, 20 };
    for (double g : gib) {
        size_t bytes = (size_t) (g * 1073741824.0);
        void* p = nullptr;
        cudaError_t se = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        if (se == cudaSuccess) {
            touch_all((char*) p, bytes);
            printf("  %5.1f GiB: OK (touched)\n", g);
            cudaFreeHost(p);
        } else {
            printf("  %5.1f GiB: FAIL %s (%s)\n", g, errname(se), cudaGetErrorString(se));
        }
    }
    print_mem_status("after-singles");

    // --- case C: cudaHostRegister on anonymous malloc pages ---
    printf("\n== C) cudaHostRegister isolation (anonymous malloc pages) ==\n");
    const size_t reg_sizes[] = { 64u*1024*1024, (size_t) 432*1024*1024 };
    for (size_t rs : reg_sizes) {
        void* q = malloc(rs);
        if (!q) { printf("  malloc %zu MiB failed\n", rs >> 20); continue; }
        touch_all((char*) q, rs);
        cudaError_t re = cudaHostRegister(q, rs, cudaHostRegisterDefault);
        if (re == cudaSuccess) {
            printf("  register %zu MiB (anonymous): OK\n", rs >> 20);
            cudaHostUnregister(q);
        } else {
            printf("  register %zu MiB (anonymous): FAIL %s (%s)\n",
                   rs >> 20, errname(re), cudaGetErrorString(re));
        }
        free(q);
    }

    // --- case D1: OS status after everything ---
    print_mem_status("end");
    printf("\nPROBE DONE\n");
    return 0;
}
