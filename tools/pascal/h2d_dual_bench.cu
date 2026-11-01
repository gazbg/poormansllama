// Plan H102: do two GPUs read host memory at twice the rate of one? Host-to-device copies of 432 MiB banks, one GPU alone, then both at once.
//   nvcc -O2 -std=c++17 tools/pascal/h2d_dual_bench.cu -o build/pascal/h2d_dual_bench
//   build/pascal/h2d_dual_bench [FILE.gguf]      (with a file: the banks are 432 MiB spans of its mapping, registered like the layer-major pass does)
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static const size_t BANK = 432ull << 20;
static const int    N    = 24;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// copies N banks, bank i taken from src[i % nsrc], two device buffers used alternately; returns seconds
static double run(int dev, const std::vector<char *> & src, std::atomic<int> * go) {
    CK(cudaSetDevice(dev));
    char * d[2];
    cudaStream_t s;
    CK(cudaMalloc(&d[0], BANK));
    CK(cudaMalloc(&d[1], BANK));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CK(cudaMemcpyAsync(d[i & 1], src[i % src.size()], BANK, cudaMemcpyHostToDevice, s));
    }
    CK(cudaStreamSynchronize(s));
    if (go != nullptr) {
        go->fetch_add(1);
        while (go->load() < 2) {
        }
    }
    const double t0 = now();
    for (int i = 0; i < N; ++i) {
        CK(cudaMemcpyAsync(d[i & 1], src[i % src.size()], BANK, cudaMemcpyHostToDevice, s));
    }
    CK(cudaStreamSynchronize(s));
    const double t = now() - t0;
    cudaFree(d[0]);
    cudaFree(d[1]);
    cudaStreamDestroy(s);
    return t;
}

static void report(const char * what, double t0, double t1) {
    const double gb = (double) N * BANK / 1e9;
    if (t1 < 0) {
        std::printf("%-46s GPU %.1f GB/s\n", what, gb / t0);
    } else {
        std::printf("%-46s GPU0 %.1f GB/s  GPU1 %.1f GB/s  total %.1f GB/s\n", what, gb / t0, gb / t1, gb / t0 + gb / t1);
    }
}

static void suite(const char * label, const std::vector<char *> & s0, const std::vector<char *> & s1) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: GPU 0 alone", label);
    report(buf, run(0, s0, nullptr), -1);
    std::snprintf(buf, sizeof buf, "%s: GPU 1 alone", label);
    report(buf, run(1, s1, nullptr), -1);
    std::atomic<int> go(0);
    double a = 0, b = 0;
    std::thread t0([&] { a = run(0, s0, &go); });
    std::thread t1([&] { b = run(1, s1, &go); });
    t0.join();
    t1.join();
    std::snprintf(buf, sizeof buf, "%s: both at once", label);
    report(buf, a, b);
}

int main(int argc, char ** argv) {
    CK(cudaSetDevice(0));
    CK(cudaFree(nullptr));
    CK(cudaSetDevice(1));
    CK(cudaFree(nullptr));
    {
        // plain pinned memory, a separate set of banks per GPU
        std::vector<char *> a, b;
        for (int i = 0; i < 4; ++i) {
            char * p;
            CK(cudaHostAlloc((void **) &p, BANK, cudaHostAllocPortable));
            std::memset(p, i + 1, BANK);
            a.push_back(p);
            CK(cudaHostAlloc((void **) &p, BANK, cudaHostAllocPortable));
            std::memset(p, i + 9, BANK);
            b.push_back(p);
        }
        suite("pinned, separate banks", a, b);
        suite("pinned, the SAME banks", a, a);
    }
    if (argc > 1) {
        const int fd = open(argv[1], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            std::fprintf(stderr, "cannot open %s\n", argv[1]);
            return 1;
        }
        char * m = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            std::fprintf(stderr, "mmap failed\n");
            return 1;
        }
        // 4 spans for each GPU, from different parts of the file; read once so that the pages are cached, then registered
        std::vector<char *> a, b;
        for (int i = 0; i < 4; ++i) {
            char * pa = m + 4096ull * 64 + (size_t) (2 * i) * BANK;
            char * pb = m + 4096ull * 64 + (size_t) (2 * i + 1) * BANK;
            volatile char sink = 0;
            for (size_t o = 0; o < BANK; o += 4096) {
                sink += pa[o] + pb[o];
            }
            (void) sink;
            CK(cudaHostRegister(pa, BANK, cudaHostRegisterPortable | cudaHostRegisterReadOnly));
            CK(cudaHostRegister(pb, BANK, cudaHostRegisterPortable | cudaHostRegisterReadOnly));
            a.push_back(pa);
            b.push_back(pb);
        }
        suite("mmap of the file, registered", a, b);
    }
    return 0;
}
