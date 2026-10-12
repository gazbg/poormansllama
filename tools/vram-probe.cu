// allocates 256 MiB chunks until cudaMalloc fails (or the limit is reached) and reports how much it got and the free memory before/after
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    const size_t limit = (size_t) (argc > 1 ? atoll(argv[1]) : 30000) << 20;
    size_t f0 = 0, t0 = 0;
    cudaMemGetInfo(&f0, &t0);
    std::vector<void *> v; size_t have = 0;
    while (have < limit) {
        void * p = nullptr;
        if (cudaMalloc(&p, 256u << 20) != cudaSuccess) { break; }
        cudaMemset(p, 1, 256u << 20);
        v.push_back(p); have += 256u << 20;
    }
    cudaDeviceSynchronize();
    size_t f1 = 0, t1 = 0;
    cudaMemGetInfo(&f1, &t1);
    printf("probe got %zu MiB (free before %zu MiB, after %zu MiB, total %zu MiB)%s\n", have >> 20, f0 >> 20, f1 >> 20, t1 >> 20, have >= limit ? " [limit reached, no failure]" : " [cudaMalloc failed]");
    return 0;
}
