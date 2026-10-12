// G22 stage B probe: head-of-line blocking of a small host-to-device copy behind a bank copy on another stream.
// Question (ledger row 56): the 256 KiB input copy that an attention layer of the layer-major prefill queues on the compute stream starts only when the
// 453 MB bank copy on the copy stream ends (nsys, g54-nsys-pp). Can the bank copy be submitted so that the small copy does not wait for all of it?
// Variants of the bank copy (432 MiB = three 144 MiB members, pinned source, device destination):
//   m3        three members, one cudaMemcpyAsync each (what the runner does today)
//   one       a single 432 MiB copy
//   cN        pieces of N MiB, ALL enqueued at once
//   pN/wW     pieces of N MiB, at most W in flight (the issuing thread waits for the event of piece i-W before issuing piece i)
// For each variant: (a) the bank copy alone (time, GB/s), (b) the latency of a 256 KiB copy on a second stream submitted `delay_us` after the bank copy
// started (pageable source like the scheduler's split inputs, and pinned source), (c) the same with the second stream at the highest priority.
// Build with tools\ce-hol-bench\build.bat; run under the GPU mutex:  powershell -NoProfile -File tools\with-gpu-lock.ps1 -VerdictTag ce-hol -Command "tools\ce-hol-bench\ce-hol-bench.exe"
//   ce-hol-bench.exe [trials=9] [delay_us=3000]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double us_between(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

static const size_t MiB = 1024ull * 1024ull;
static const size_t BANK = 432 * MiB;
static const size_t MEMBER = 144 * MiB;
static const size_t SMALL = 256 * 1024;

struct variant {
    std::string name;
    int kind;           // 0 = members, 1 = single, 2 = pieces unpaced, 3 = pieces paced
    size_t piece_mib;
    int window;
};

static char * g_h_bank = nullptr;
static char * g_d_bank = nullptr;
static cudaStream_t g_copy, g_comp, g_comp_hi;
static std::vector<cudaEvent_t> g_ring;

// issue the whole bank on g_copy; returns when everything is ISSUED (paced variants block here, which is why they run on a worker thread)
static void issue_bank(const variant & v) {
    if (v.kind == 0) {
        for (int m = 0; m < 3; ++m) {
            CK(cudaMemcpyAsync(g_d_bank + m * MEMBER, g_h_bank + m * MEMBER, MEMBER, cudaMemcpyHostToDevice, g_copy));
        }
    } else if (v.kind == 1) {
        CK(cudaMemcpyAsync(g_d_bank, g_h_bank, BANK, cudaMemcpyHostToDevice, g_copy));
    } else {
        const size_t p = v.piece_mib * MiB;
        const size_t n = (BANK + p - 1) / p;
        for (size_t i = 0; i < n; ++i) {
            if (v.kind == 3 && i >= (size_t) v.window) {
                CK(cudaEventSynchronize(g_ring[i % v.window]));
            }
            const size_t off = i * p;
            const size_t len = std::min(p, BANK - off);
            CK(cudaMemcpyAsync(g_d_bank + off, g_h_bank + off, len, cudaMemcpyHostToDevice, g_copy));
            if (v.kind == 3) {
                CK(cudaEventRecord(g_ring[i % v.window], g_copy));
            }
        }
    }
}

struct result {
    double bank_us = 0.0;     // start of issue -> bank copy complete
    double small_us = 0.0;    // submit of the small copy -> its completion (host view)
};

// one trial. small_src == nullptr: bank alone. prio: use the high-priority stream for the small copy.
static result trial(const variant & v, void * small_src, void * d_small, bool prio, int delay_us) {
    CK(cudaStreamSynchronize(g_copy));
    CK(cudaStreamSynchronize(g_comp));
    CK(cudaStreamSynchronize(g_comp_hi));
    result r;
    const auto t0 = clk::now();
    std::thread worker([&] { issue_bank(v); });
    if (small_src != nullptr) {
        while (us_between(t0, clk::now()) < (double) delay_us) { /* spin */ }
        cudaStream_t s = prio ? g_comp_hi : g_comp;
        const auto ts0 = clk::now();
        CK(cudaMemcpyAsync(d_small, small_src, SMALL, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));
        r.small_us = us_between(ts0, clk::now());
    }
    worker.join();
    CK(cudaStreamSynchronize(g_copy));
    r.bank_us = us_between(t0, clk::now());
    return r;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    const int trials   = argc > 1 ? atoi(argv[1]) : 9;
    const int delay_us = argc > 2 ? atoi(argv[2]) : 3000;
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    int lo = 0, hi = 0;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    printf("device: %s, asyncEngineCount %d, stream priority range [%d (lowest), %d (highest)], trials %d, small copy %d us after the bank copy starts\n",
           prop.name, prop.asyncEngineCount, lo, hi, trials, delay_us);

    CK(cudaHostAlloc((void **) &g_h_bank, BANK, cudaHostAllocDefault));
    memset(g_h_bank, 0x5a, BANK);
    CK(cudaMalloc((void **) &g_d_bank, BANK));
    void * d_small = nullptr;
    CK(cudaMalloc(&d_small, SMALL));
    void * h_pageable = malloc(SMALL);
    memset(h_pageable, 1, SMALL);
    void * h_pinned = nullptr;
    CK(cudaHostAlloc(&h_pinned, SMALL, cudaHostAllocDefault));
    memset(h_pinned, 2, SMALL);
    CK(cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&g_comp, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithPriority(&g_comp_hi, cudaStreamNonBlocking, hi));
    g_ring.resize(64);
    for (auto & e : g_ring) {
        CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }

    // reference: the small copy alone
    {
        std::vector<double> a, b;
        for (int i = 0; i < trials; ++i) {
            CK(cudaStreamSynchronize(g_comp));
            auto t = clk::now();
            CK(cudaMemcpyAsync(d_small, h_pageable, SMALL, cudaMemcpyHostToDevice, g_comp));
            CK(cudaStreamSynchronize(g_comp));
            a.push_back(us_between(t, clk::now()));
            t = clk::now();
            CK(cudaMemcpyAsync(d_small, h_pinned, SMALL, cudaMemcpyHostToDevice, g_comp));
            CK(cudaStreamSynchronize(g_comp));
            b.push_back(us_between(t, clk::now()));
        }
        printf("small copy alone (256 KiB): pageable %.1f us, pinned %.1f us (medians)\n\n", median(a), median(b));
    }

    std::vector<variant> vs = {
        {"m3 (3 x 144 MiB, today)", 0, 0, 0},
        {"one (432 MiB)", 1, 0, 0},
        {"c1  (1 MiB, unpaced)", 2, 1, 0},
        {"c2  (2 MiB, unpaced)", 2, 2, 0},
        {"c4  (4 MiB, unpaced)", 2, 4, 0},
        {"c8  (8 MiB, unpaced)", 2, 8, 0},
        {"c16 (16 MiB, unpaced)", 2, 16, 0},
        {"c32 (32 MiB, unpaced)", 2, 32, 0},
        {"p2/w2", 3, 2, 2},
        {"p2/w4", 3, 2, 4},
        {"p4/w2", 3, 4, 2},
        {"p4/w3", 3, 4, 3},
        {"p4/w4", 3, 4, 4},
        {"p8/w2", 3, 8, 2},
        {"p8/w3", 3, 8, 3},
        {"p16/w2", 3, 16, 2},
    };
    printf("%-26s | %-10s %-8s | %-27s | %-27s | %-27s\n", "variant", "bank ms", "GB/s", "small pageable us (med/min/max)", "small pinned us (med/min/max)", "pageable, high prio us");
    for (const auto & v : vs) {
        // warm one pass (page tables, driver paths)
        trial(v, nullptr, d_small, false, delay_us);
        std::vector<double> alone;
        for (int i = 0; i < trials; ++i) {
            alone.push_back(trial(v, nullptr, d_small, false, delay_us).bank_us);
        }
        std::vector<double> sp, sn, sh, bp;
        for (int i = 0; i < trials; ++i) {
            const result r = trial(v, h_pageable, d_small, false, delay_us);
            sp.push_back(r.small_us);
            bp.push_back(r.bank_us);
            sn.push_back(trial(v, h_pinned, d_small, false, delay_us).small_us);
            sh.push_back(trial(v, h_pageable, d_small, true, delay_us).small_us);
        }
        auto mm = [](const std::vector<double> & x) {
            char buf[96];
            snprintf(buf, sizeof(buf), "%8.0f %8.0f %8.0f", median(x), *std::min_element(x.begin(), x.end()), *std::max_element(x.begin(), x.end()));
            return std::string(buf);
        };
        const double bank_ms = median(alone) / 1000.0;
        printf("%-26s | %8.2f %8.2f | %s | %s | %s   (bank with the small copy: %.2f ms)\n", v.name.c_str(), bank_ms, (double) BANK / 1e9 / (bank_ms / 1000.0),
               mm(sp).c_str(), mm(sn).c_str(), mm(sh).c_str(), median(bp) / 1000.0);
        fflush(stdout);
    }
    return 0;
}
