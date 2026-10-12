// Holds N MiB of device memory so that a process under test sees exactly (free - N) MiB through cudaMemGetInfo / cudaMalloc: the owner's llama-server
// flag --fit-target 16000 ("leave 16000 MiB free on the 24 GiB card to emulate 8 GB") is the same constraint stated from the other side.
//   vram-ballast.exe <MiB> [chunk MiB]     prints "ballast ready <MiB> free_before <MiB> free_after <MiB>" and then waits until stdin is closed or it is killed.
// Memory is allocated in chunks (default 512 MiB) and every chunk is memset so the pages really exist.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    const size_t want  = (size_t) (argc > 1 ? atoll(argv[1]) : 0) << 20;
    const size_t chunk = (size_t) (argc > 2 ? atoll(argv[2]) : 512) << 20;
    size_t f0 = 0, t0 = 0;
    if (cudaMemGetInfo(&f0, &t0) != cudaSuccess) { fprintf(stderr, "cudaMemGetInfo failed\n"); return 1; }
    std::vector<void *> blocks;
    size_t have = 0;
    while (have < want) {
        const size_t n = (want - have) < chunk ? (want - have) : chunk;
        void * p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) { fprintf(stderr, "cudaMalloc of %zu MiB failed after %zu MiB\n", n >> 20, have >> 20); return 2; }
        cudaMemset(p, 0, n);
        blocks.push_back(p);
        have += n;
    }
    cudaDeviceSynchronize();
    size_t f1 = 0, t1 = 0;
    cudaMemGetInfo(&f1, &t1);
    printf("ballast ready %zu MiB free_before %zu MiB free_after %zu MiB total %zu MiB\n", have >> 20, f0 >> 20, f1 >> 20, t1 >> 20);
    fflush(stdout);
    int c;
    while ((c = getchar()) != EOF) { }   // parent closes stdin (or kills us) to release the memory
    return 0;
}
