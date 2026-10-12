// G23 row 63 probe: the PCIe link of the laptop GPU downshifts (Gen4 -> Gen2 -> Gen1: a 32 MiB host-to-device copy takes 2.35 -> 9.2 -> 18.4 ms) while the GPU idles under a CPU-bound run
// (nvidia-smi, g81-pcie). Which kind of GPU activity keeps the link at Gen4?  Every 100 ms the program times one 32 MiB pinned host-to-device copy (cudaEvent timing) while a background
// thread keeps the GPU "alive" in one of these ways:
//   mode 0  nothing (the control: the downshift should appear)
//   mode 1  a tiny kernel every 1 ms (launch only: ~5 us of GPU time)
//   mode 2  a kernel that spins `busy_us` microseconds every `period_us` microseconds (a duty cycle)
//   mode 3  a kernel that spins continuously (100% utilization)
//   mode 4  a 4 KiB host-to-device copy every 1 ms (link traffic without compute)
//   mode 5  wake-up test: idle for `seconds`, then 80 back-to-back 32 MiB copies timed one by one
// The copy time classifies the link: ~2.3 ms Gen4, ~9.2 ms Gen2, ~18.4 ms Gen1 (x8). Prints a line per second: median copy time of that second and the state counts.
// Build with tools\pcie-state-bench\build.bat; run under the GPU mutex:  powershell -NoProfile -File tools\with-gpu-lock.ps1 -VerdictTag pcie -Command "tools\pcie-state-bench\pcie-state-bench.exe 0 40"
//   pcie-state-bench.exe [mode=0] [seconds=40] [period_us=1000] [busy_us=200]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;

__global__ void tiny(float * p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[0] += 1.0f;
    }
}

__global__ void spin(float * p, long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[0] += 1.0f;
    }
}

int main(int argc, char ** argv) {
    const int mode      = argc > 1 ? atoi(argv[1]) : 0;
    const int seconds   = argc > 2 ? atoi(argv[2]) : 40;
    const int period_us = argc > 3 ? atoi(argv[3]) : 1000;
    const int busy_us   = argc > 4 ? atoi(argv[4]) : 200;
    CK(cudaSetDevice(0));
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const size_t N = 32ull << 20;
    char * h = nullptr;
    CK(cudaHostAlloc((void **) &h, N, cudaHostAllocDefault));
    char * d = nullptr;
    CK(cudaMalloc((void **) &d, N));
    float * f = nullptr;
    CK(cudaMalloc((void **) &f, 64));
    CK(cudaMemset(f, 0, 64));
    char * hs = nullptr;
    CK(cudaHostAlloc((void **) &hs, 4096, cudaHostAllocDefault));
    char * ds = nullptr;
    CK(cudaMalloc((void **) &ds, 4096));
    cudaStream_t s_main, s_bg;
    CK(cudaStreamCreateWithFlags(&s_main, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::atomic<bool> stop{false};
    std::thread bg;
    if (mode != 0 && mode != 5) {
        bg = std::thread([&] {
            const long long cycles = (long long) clock_khz * busy_us / 1000;
            while (!stop.load()) {
                const auto t = clk::now();
                if (mode == 1) {
                    tiny<<<1, 32, 0, s_bg>>>(f);
                } else if (mode == 2) {
                    spin<<<1, 32, 0, s_bg>>>(f, cycles);
                } else if (mode == 3) {
                    spin<<<1, 32, 0, s_bg>>>(f, (long long) clock_khz * 5);    // 5 ms per launch, back to back
                } else if (mode == 4) {
                    cudaMemcpyAsync(ds, hs, 4096, cudaMemcpyHostToDevice, s_bg);
                }
                if (mode == 3) {
                    cudaStreamSynchronize(s_bg);
                    continue;
                }
                while (std::chrono::duration<double, std::micro>(clk::now() - t).count() < (mode == 2 ? period_us : 1000)) {
                }
            }
            cudaStreamSynchronize(s_bg);
        });
    }
    if (mode == 5) {
        // wake-up test: the GPU idles for `seconds` (no copies, no kernels), then 80 back-to-back 32 MiB copies are timed one by one; the series shows how long the link takes to climb back to Gen4
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        printf("mode 5: after %d s of idle, 80 back-to-back 32 MiB copies (ms each; ~2.3 = Gen4, ~9.2 = Gen2, ~18.4 = Gen1):\n", seconds);
        double t_acc = 0.0;
        for (int i = 0; i < 80; ++i) {
            CK(cudaEventRecord(e0, s_main));
            CK(cudaMemcpyAsync(d, h, N, cudaMemcpyHostToDevice, s_main));
            CK(cudaEventRecord(e1, s_main));
            CK(cudaEventSynchronize(e1));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            t_acc += ms;
            printf("%s%.2f%s", i % 10 == 0 ? "  " : "", ms, i % 10 == 9 ? "\n" : " ");
        }
        printf("  total %.1f ms for 80 x 32 MiB = %.2f GB/s average\n", t_acc, 80.0 * 33.554432 / t_acc);
        return 0;
    }
    printf("mode %d, %d s, period %d us, busy %d us (SM clock %d kHz)\n", mode, seconds, period_us, busy_us, clock_khz);
    const auto t_start = clk::now();
    std::vector<double> sec;
    int cur = 0;
    while (true) {
        const double t = std::chrono::duration<double>(clk::now() - t_start).count();
        if (t >= seconds) {
            break;
        }
        CK(cudaEventRecord(e0, s_main));
        CK(cudaMemcpyAsync(d, h, N, cudaMemcpyHostToDevice, s_main));
        CK(cudaEventRecord(e1, s_main));
        CK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if ((int) t > cur) {
            std::sort(sec.begin(), sec.end());
            int g4 = 0, g2 = 0, g1 = 0;
            for (double x : sec) {
                (x < 4.0 ? g4 : x < 13.0 ? g2 : g1)++;
            }
            printf("t=%3d s  copy median %6.2f ms   (Gen4-like %d, Gen2-like %d, Gen1-like %d)\n", cur, sec.empty() ? 0.0 : sec[sec.size() / 2], g4, g2, g1);
            fflush(stdout);
            sec.clear();
            cur = (int) t;
        }
        sec.push_back(ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    stop = true;
    if (bg.joinable()) {
        bg.join();
    }
    return 0;
}
