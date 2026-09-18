// Relocated from build/gplr32-enginecount.cu (2026-09-16 provenance sweep; the T-GPLR-34
// convention: committed runners/probes live under tools/, with paths made
// relative to the repository root so a clean checkout reproduces the run).
// Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
// T-GPLR-32 (research support): print asyncEngineCount for device 0.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n < 1) { printf("no device: %s\n", cudaGetErrorName(e)); return 1; }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    printf("device: %s\n", p.name);
    printf("asyncEngineCount: %d (0 = copies NOT concurrent with compute; 1 = one copy engine; 2 = full duplex)\n",
           p.asyncEngineCount);
    printf("concurrentKernels: %d\n", p.concurrentKernels);
    return 0;
}
