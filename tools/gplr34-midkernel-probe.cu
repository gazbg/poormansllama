// T-GPLR-34: standalone mid-kernel H2D chaining probe (rev 4, final).
// Runner: tools/run-gplr34-probe.ps1 (nvcc -O2 -arch=native, then the
// run under tools/with-gpu-lock.ps1). -arch=native is REQUIRED on this
// box: the default nvcc target's PTX is JIT-rejected on sm_120 at
// kernel launch. Recorded results: docs/currentstate.md, 2026-09-16.
//
// Question: when a bank-sized pinned H2D copy is enqueued while a compute
// kernel is still running, does the transfer overlap the kernel?
//
// rev 2 finding: single-stream mid-kernel enqueue is additive
// (62.24 = 30.74 + 31.37) — but that is guaranteed by stream ordering on
// every platform, so it does not test cross-engine behavior.
// rev 4 final: arm [6] runs the kernel on stream s and enqueues the bank
// copy on stream s2 MID-kernel, with cross-stream event timestamps:
//   t_start = elapsed(kernel-start, copy-start marker on s2)
//             ~0 ms => CE started immediately (concurrent engines)
//             ~k  ms => copy waited for kernel end (serialized)
//   t_cwin  = the copy's own window (31.4 both ways; sanity)
//   t_total = kernel-start -> copy-done (31.4 / ~62)
//   t_kwin  = the kernel's own window (stretched by concurrent CE?)
//
// Bank bytes match the Qwen3.6-35B-A3B routed-expert bank (3 x 144 MiB).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

static const size_t BANK  = 452984832ull;  // 452,984,832 = 3 * 150,994,944
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  exit(1); } } while (0)
#define GL(stmt) do { (stmt); cudaError_t le_ = cudaGetLastError(); if (le_ != cudaSuccess) { \
  printf("LAUNCH ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(le_)); \
  exit(1); } } while (0)

// Keep-alive burn kernel: dependent FMA chain, result stored. n <= threads,
// so each active thread runs the full iters chain once.
__global__ void burn(float * buf, size_t n, int iters, int salt) {
  const size_t i = threadIdx.x + blockIdx.x * (size_t)blockDim.x;
  if (i >= n) return;
  float acc = salt * 0.001f + (float)i * 1e-9f;
  for (int k = 0; k < iters; ++k) acc = fmaf(acc, 1.0000001f, 1e-8f);
  buf[i] = acc;
}

struct Stats { float minv, mean; };
static Stats summarize(const float * v, int n) {
  float s = 0.0f, m = v[0];
  for (int i = 0; i < n; ++i) { s += v[i]; m = std::min(m, v[i]); }
  return { m, s / n };
}

int main() {
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("bank bytes: %zu\n", BANK);

  float * src = nullptr; CK(cudaHostAlloc((void **)&src, BANK, cudaHostAllocDefault));
  for (size_t i = 0; i < BANK / sizeof(float); ++i) src[i] = (float)i;
  float * dst = nullptr; CK(cudaMalloc((void **)&dst, BANK));
  float * kbuf = nullptr; CK(cudaMalloc((void **)&kbuf, 132 * 256 * sizeof(float)));
  CK(cudaMemset(kbuf, 0, 132 * 256 * sizeof(float)));

  cudaStream_t s;  CK(cudaStreamCreate(&s));   // "compute" stream
  cudaStream_t s2; CK(cudaStreamCreate(&s2));  // "copy" stream
  cudaEvent_t e0, e1, e2, e3; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
  cudaEvent_t kdone; CK(cudaEventCreate(&kdone));

  // warm-up: driver + clocks
  for (int i = 0; i < 5; ++i) {
    CK(cudaMemcpyAsync(dst, src, BANK, cudaMemcpyHostToDevice, s));
    GL((burn<<<132, 256, 0, s>>>(kbuf, 132 * 256, 1000, i)));
  }
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamSynchronize(s2));

  const int N = 10;

  // ---- calibrate burn kernel to ~34 ms, with a loud failure if it cannot ----
  float t0 = 0.0f; int iters = 100000;
  for (int round = 0; round < 6; ++round) {
    CK(cudaEventRecord(e0, s));
    GL((burn<<<132, 256, 0, s>>>(kbuf, 132 * 256, iters, 99)));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&t0, e0, e1));
    printf("cal round %d: iters %d -> %8.3f ms\n", round, iters, t0);
    if (t0 >= 30.0f && t0 <= 40.0f) break;
    if (t0 < 0.05f) { printf("FATAL: burn kernel measures ~0 ms; instrument broken\n"); return 2; }
    iters = (int)((double)iters * 34.0 / std::max(0.5, (double)t0));
    if (iters > 200000000) { printf("FATAL: iters overflow guard\n"); return 2; }
  }
  if (t0 < 20.0f) { printf("FATAL: calibration never reached >=20 ms\n"); return 2; }

  // ---- [1] pure bank copy (baseline) ----
  float v1[N];
  for (int i = 0; i < N; ++i) {
    CK(cudaEventRecord(e0, s));
    CK(cudaMemcpyAsync(dst, src, BANK, cudaMemcpyHostToDevice, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&v1[i], e0, e1));
  }
  Stats s1 = summarize(v1, N);
  printf("[1] pure copy      : min %8.2f ms  mean %8.2f ms  (%.2f GB/s)\n",
         s1.minv, s1.mean, BANK / (s1.minv * 1e6));

  // ---- [5] single-stream mid-kernel enqueue (stream-ordering reference) ----
  float g5[N];
  for (int i = 0; i < N; ++i) {
    CK(cudaEventRecord(e0, s));
    GL((burn<<<132, 256, 0, s>>>(kbuf, 132 * 256, iters, i + 100)));
    CK(cudaMemcpyAsync(dst, src, BANK, cudaMemcpyHostToDevice, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&g5[i], e0, e1));
  }
  Stats s5 = summarize(g5, N);
  printf("[5] 1-stream k+copy: min %8.2f ms  mean %8.2f ms  (k+copy expected: stream ordering)\n",
         s5.minv, s5.mean);

  // ---- [6] TWO-STREAM: kernel on s, bank copy on s2 enqueued mid-kernel ----
  float t_start[N], t_cwin[N], t_total[N], t_kwin[N];
  for (int i = 0; i < N; ++i) {
    CK(cudaEventRecord(e0, s));
    GL((burn<<<132, 256, 0, s>>>(kbuf, 132 * 256, iters, i + 200)));
    CK(cudaEventRecord(e2, s2));                                    // copy-start marker
    CK(cudaMemcpyAsync(dst, src, BANK, cudaMemcpyHostToDevice, s2));
    CK(cudaEventRecord(e3, s2));                                    // copy-done marker
    CK(cudaEventRecord(kdone, s));
    CK(cudaEventSynchronize(e3));
    CK(cudaEventSynchronize(kdone));
    CK(cudaEventElapsedTime(&t_start[i], e0, e2));  // kernel start -> copy start
    CK(cudaEventElapsedTime(&t_cwin[i], e2, e3));   // copy's own window
    CK(cudaEventElapsedTime(&t_total[i], e0, e3));  // kernel start -> copy done
    CK(cudaEventElapsedTime(&t_kwin[i], e0, kdone)); // kernel's own window
  }
  Stats sst = summarize(t_start, N), scw = summarize(t_cwin, N);
  Stats stt = summarize(t_total, N), skw = summarize(t_kwin, N);
  printf("[6] 2-stream copy  : copy START after kernel-start: min %6.2f mean %6.2f ms  (0=concurrent, ~k=serialized)\n",
         sst.minv, sst.mean);
  printf("                     copy window                     : min %6.2f mean %6.2f ms  (31.4 sanity)\n",
         scw.minv, scw.mean);
  printf("                     kernel-start -> copy-done       : min %6.2f mean %6.2f ms  (31.4=concurrent, ~62=serialized)\n",
         stt.minv, stt.mean);
  printf("                     kernel window                   : min %6.2f mean %6.2f ms  (30.7=unaffected)\n",
         skw.minv, skw.mean);
  printf("PROBE DONE\n");
  return 0;
}
