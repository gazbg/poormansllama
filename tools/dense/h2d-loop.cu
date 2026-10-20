// Host-to-device copy load for a fixed time: the copy-engine traffic of the decode ring, without any model.
//   nvcc -O2 -arch=sm_61 -o build/dense/h2d-loop tools/dense/h2d-loop.cu
//   CUDA_VISIBLE_DEVICES=0 build/dense/h2d-loop SECONDS [CHUNK_MIB]
// Copies a page-locked 1 GiB buffer to the device in CHUNK_MIB pieces (default 32) back to back and prints the achieved GB/s.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char ** argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 10.0;
    size_t chunk = (argc > 2 ? atoi(argv[2]) : 32) * (size_t) 1048576;
    size_t total = (size_t) 1 << 30;
    void * host; void * dev;
    CK(cudaSetDevice(0));
    CK(cudaMallocHost(&host, total));
    for (size_t i = 0; i < total; i += 4096) ((char *) host)[i] = 1;   // touch every page
    CK(cudaMalloc(&dev, chunk));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto t0 = std::chrono::steady_clock::now();
    size_t bytes = 0, off = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        for (int i = 0; i < 8; i++) { CK(cudaMemcpyAsync(dev, (char *) host + off, chunk, cudaMemcpyHostToDevice, s)); off = (off + chunk) % total; bytes += chunk; }
        CK(cudaStreamSynchronize(s));
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("h2d-loop: %.2f GB/s over %.1f s (chunk %zu MiB)\n", bytes / 1e9 / dt, dt, chunk >> 20);
    return 0;
}
